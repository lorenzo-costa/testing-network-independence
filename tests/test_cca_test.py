import numpy as np
import pytest

from src.load_config import _resolve_method
from src.load_config import build_factorial_design, load_config
from src.helper_functions.simulation_functions import run_scenario
from src.methods import CanonicalCorrelationTest, ConditionalGenerator
from src.metrics import ReturnMetric
from src.test_functions import first_cca_component
from pathlib import Path


def test_cca_statistic_matches_absolute_correlation_and_constant_case():
    z = np.array([0.0, 1.0, 3.0, 4.0])[:, None]
    y = np.array([1.0, 4.0, 2.0, 7.0])[:, None]
    assert first_cca_component(z, y) == pytest.approx(abs(np.corrcoef(z[:, 0], y[:, 0])[0, 1]))
    assert first_cca_component(z, np.ones_like(y)) == 0.0


def test_cca_model_x_crt_resamples_y_conditionally_and_is_reproducible():
    class ConditionalY(ConditionalGenerator):
        def __init__(self):
            self.seen_x = []

        def sample(self, X):
            self.seen_x.append(X.copy())
            return X + self.rng.normal(size=X.shape)

    rng = np.random.default_rng(25)
    data = {
        "Z": rng.normal(size=(20, 2)),
        "X": rng.normal(size=(20, 1)),
        "Y": rng.normal(size=(20, 1)),
    }
    results = []
    for _ in range(2):
        generator = ConditionalY()
        method = CanonicalCorrelationTest(
            use_true_latent=True, test_type="model_X",
            conditional_generator=generator, npermutations=9,
            rng=np.random.default_rng(4),
        )
        method.fit(data)
        assert method.get_name() == "CCA_ModelXTest_covariate"
        assert method.test_stat_estimate == pytest.approx(first_cca_component(data["Z"], data["Y"]))
        assert len(generator.seen_x) == 9
        assert all(np.array_equal(x, data["X"]) for x in generator.seen_x)
        results.append(method)
    assert results[0].null_distribution == pytest.approx(results[1].null_distribution)
    assert results[0].pvalue == results[1].pvalue


def test_cca_config_resolves_model_x():
    method = _resolve_method({
        "name": "CanonicalCorrelationTest",
        "kwargs": {"test_type": "model_X", "conditional_generator": "from_dgp"},
    })
    assert method.func is CanonicalCorrelationTest
    assert method.keywords == {"test_type": "model_X", "conditional_generator": "from_dgp"}


def test_cca_crt_runs_through_yaml_design_and_dgp_generator():
    config = load_config(Path(__file__).resolve().parents[1] / "config_rdpg_stratum.yaml")
    row = build_factorial_design(config)[0].copy()
    row.update(n=20, npermutations=2, use_true_latent=True)
    row["method"] = _resolve_method({
        "name": "CanonicalCorrelationTest",
        "kwargs": {"test_type": "model_X", "conditional_generator": "from_dgp"},
    })
    result = run_scenario([ReturnMetric()], row, seed=np.random.SeedSequence(7))
    assert result["args"]["method_name"] == "CCA_ModelXTest_covariate"
    assert 0 <= result["ReturnMetric"]["p-value"] <= 1
