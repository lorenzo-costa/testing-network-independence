import numpy as np
import pytest

from src.methods import (
    GaussianKDEConditionalEstimator, CanonicalCorrelationTest, MultivariateACTest,
)


def test_conditional_density_and_samples_depend_on_x():
    rng = np.random.default_rng(10)
    x = rng.uniform(-1, 1, size=(300, 1))
    y = x + rng.normal(scale=0.1, size=(300, 1))
    estimator = GaussianKDEConditionalEstimator(
        bandwidth=[0.1, 0.3], n_splits=3, rng=np.random.default_rng(11)
    ).fit(x, y)
    queries = np.vstack([np.full((1000, 1), -0.8), np.full((1000, 1), 0.8)])
    draws = estimator.sample(queries)
    assert draws.shape == (2000, 1)
    assert draws[:1000].mean() < -0.5
    assert draws[1000:].mean() > 0.5
    assert estimator.pdf(np.array([[-0.8], [0.8]]), np.array([[-0.8], [0.8]])).shape == (2,)
    assert estimator.bandwidth_ in (0.1, 0.3)


def test_model_x_estimated_conditional_uses_held_out_rows():
    rng = np.random.default_rng(12)
    x = rng.uniform(size=(60, 1))
    y = x + rng.normal(scale=0.1, size=(60, 1))
    z = rng.normal(size=(60, 1))
    estimator = GaussianKDEConditionalEstimator(bandwidth=0.2)
    test = CanonicalCorrelationTest(
        k=1, use_true_latent=True, test_type="model_X",
        estimate_conditional=True, conditional_estimator=estimator,
        test_size=0.4, npermutations=9, rng=np.random.default_rng(13),
    )
    test.fit({"Z": z, "Y": y, "X": x})
    assert len(test.train_indices) == 36
    assert len(test.test_indices) == 24
    assert set(test.train_indices).isdisjoint(test.test_indices)
    np.testing.assert_array_equal(estimator.x_train_, x[test.train_indices, 0])
    assert test._draw_null_data().shape == (24, 1)


def test_scalar_inputs_are_required():
    estimator = GaussianKDEConditionalEstimator()
    with pytest.raises(ValueError, match="scalar column"):
        estimator.fit(np.ones((10, 2)), np.ones((10, 1)))


def test_ac_embeds_full_graph_then_calibrates_only_test_nodes(monkeypatch):
    rng = np.random.default_rng(22)
    n = 24
    x = np.arange(n, dtype=float)[:, None] / n
    y = x + rng.normal(scale=0.1, size=(n, 1))
    z = rng.normal(size=(n, 1))
    adjacency_shapes = []
    statistic_inputs = []

    def solver(A, k, rng):
        adjacency_shapes.append(A.shape)
        return z.copy(), None

    def statistic(*, Y, Z, X, **kwargs):
        statistic_inputs.append((Y.copy(), Z.copy(), X.copy()))
        return float(np.mean(Y * X))

    monkeypatch.setattr("src.methods.ac_test.ac_coefficient", statistic)
    test = MultivariateACTest(
        k=1, solver=solver, M=1, standardize=True, test_type="model_X",
        estimate_conditional=True,
        conditional_estimator=GaussianKDEConditionalEstimator(bandwidth=0.2),
        npermutations=3, rng=np.random.default_rng(23),
    )
    test.fit({"A": np.zeros((n, n)), "X": x, "Y": y})
    assert adjacency_shapes == [(n, n)]
    assert len(statistic_inputs) == 4
    expected_scale = np.sqrt(np.mean(np.var(z[test.test_indices], axis=0)))
    assert test.standardization_scale_ == pytest.approx(expected_scale)
    for tested_y, tested_z, tested_x in statistic_inputs:
        assert tested_y.shape == tested_z.shape == tested_x.shape == (12, 1)
        np.testing.assert_array_equal(tested_z, z[test.test_indices] / expected_scale)
        np.testing.assert_array_equal(tested_x, x[test.test_indices])
    np.testing.assert_array_equal(statistic_inputs[0][0], y[test.test_indices])
