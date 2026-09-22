import numpy as np
import pytest
from scipy.special import expit

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.latent_samplers import MultipleNetworksSampler


@pytest.mark.parametrize(
    "network_class,options",
    [(GaussianNetwork, {}), (BernoulliNetwork, {}), (BernoulliNetwork, {"rdpg": True})],
)
def test_networks_accept_zero_B_shorthand(network_class, options):
    kwargs = dict(n=7, p=3, d_x=2, d_y=4, x_mean=0.1, x_variance=0, eps_variance=0)
    shorthand = network_class(**kwargs, **options, B=0, rng=np.random.default_rng(82))
    explicit = network_class(
        **kwargs, **options, B=np.zeros((4, 6)), rng=np.random.default_rng(82)
    )
    for _ in range(2):
        actual, expected = shorthand.generate(), explicit.generate()
        np.testing.assert_array_equal(actual["B"], np.zeros((4, 6)))
        np.testing.assert_array_equal(actual["Y"], np.zeros((7, 4)))
        for key in ("A_Y", "Y", "B"):
            np.testing.assert_array_equal(actual[key], expected[key])
        for key in ("A_X", "X"):
            for a, b in zip(actual[key], expected[key]):
                np.testing.assert_array_equal(a, b)
        assert shorthand.rng.bit_generator.state == explicit.rng.bit_generator.state


@pytest.mark.parametrize(
    "n,p,d_x,d_y", [(12, 3, 2, 4), (1, 1, 1, 1), (1, 3, 2, 1), (7, 2, 1, 3)]
)
def test_gaussian_network_contract(n, p, d_x, d_y):
    network = GaussianNetwork(n, p, d_x, d_y, rng=np.random.default_rng(1))
    result = network.generate()

    assert set(result) == {"A_Y", "A_X", "Y", "X", "B"}
    assert isinstance(result["X"], list) and len(result["X"]) == p
    assert isinstance(result["A_X"], list) and len(result["A_X"]) == p
    assert all(isinstance(x, np.ndarray) and x.shape == (n, d_x) for x in result["X"])
    assert result["Y"].shape == (n, d_y)
    assert result["B"].shape == (d_y, p * d_x)
    for adjacency in [result["A_Y"], *result["A_X"]]:
        assert isinstance(adjacency, np.ndarray)
        assert adjacency.shape == (n, n)
        np.testing.assert_array_equal(adjacency, adjacency.T)
        np.testing.assert_array_equal(adjacency.diagonal(), np.zeros(n))


@pytest.mark.parametrize("network_class", [GaussianNetwork, BernoulliNetwork])
def test_networks_forward_active_B_network_fraction(network_class):
    network = network_class(
        4,
        4,
        2,
        3,
        b_mean=1,
        b_variance=0,
        b_active_network_fraction=0.5,
        rng=np.random.default_rng(93),
    )

    for _ in range(2):
        blocks = np.split(network.generate()["B"], 4, axis=1)
        assert sum(np.any(block) for block in blocks) == 2


def test_gaussian_network_zero_noise_matches_latent_model_and_gram_matrices():
    B = np.array([[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.0, 1.0]])
    result = GaussianNetwork(
        8, 2, 2, 2, B=B, eps_variance=0, edge_var=0, rng=np.random.default_rng(2)
    ).generate()

    np.testing.assert_array_equal(result["B"], B)
    np.testing.assert_array_equal(
        result["Y"], np.concatenate(result["X"], axis=1) @ B.T
    )
    for adjacency, latent in zip(
        [result["A_Y"], *result["A_X"]], [result["Y"], *result["X"]]
    ):
        expected = latent @ latent.T
        np.fill_diagonal(expected, 0)
        np.testing.assert_array_equal(adjacency, expected)


def test_gaussian_network_uses_shared_rng_and_mirrors_exact_upper_triangle():
    rng, reference_rng = np.random.default_rng(3), np.random.default_rng(3)
    kwargs = dict(n=6, p=2, d_x=2, d_y=3, x_variance=4, eps_variance=9, b_variance=2)
    network = GaussianNetwork(**kwargs, edge_var=16, rng=rng)
    assert network.rng is rng
    assert network.latent_sampler.rng is rng
    result = network.generate()

    reference = MultipleNetworksSampler(**kwargs, rng=reference_rng).sample_latent()
    np.testing.assert_array_equal(result["B"], reference["B"])
    for actual, expected in zip(
        [result["Y"], *result["X"]], [reference["Y"], *reference["X"]]
    ):
        np.testing.assert_array_equal(actual, expected)
    for adjacency, latent in zip(
        [result["A_Y"], *result["A_X"]], [reference["Y"], *reference["X"]]
    ):
        sampled = reference_rng.normal(loc=latent @ latent.T, scale=4)
        upper = np.triu(sampled, k=1)
        np.testing.assert_array_equal(adjacency, upper + upper.T)
        assert not np.allclose(adjacency, (sampled + sampled.T) / 2)
    assert rng.bit_generator.state == reference_rng.bit_generator.state


def test_gaussian_network_reproducibility_across_repeated_generations():
    kwargs = dict(n=5, p=3, d_x=2, d_y=4, x_mean=1, edge_var=0.5)
    a = GaussianNetwork(**kwargs, rng=np.random.default_rng(4))
    b = GaussianNetwork(**kwargs, rng=np.random.default_rng(4))
    for _ in range(2):
        result_a, result_b = a.generate(), b.generate()
        for key in ("A_Y", "Y", "B"):
            np.testing.assert_array_equal(result_a[key], result_b[key])
        for key in ("A_X", "X"):
            for array_a, array_b in zip(result_a[key], result_b[key]):
                np.testing.assert_array_equal(array_a, array_b)


def test_gaussian_network_forwards_covariances_and_distribution_choices():
    kwargs = dict(
        n=4,
        p=2,
        d_x=1,
        d_y=2,
        B=np.eye(2),
        x_mean=[1.0, 2.0],
        x_variance=[[2.0, 0.7], [0.7, 1.0]],
        eps_variance=[[1.0, -0.2], [-0.2, 1.0]],
        x_distribution="multivariate_gaussian",
        eps_distribution="multivariate_gaussian",
        b_distribution="multivariate_gaussian",
        b_mean=2,
        b_variance=3,
    )
    network = GaussianNetwork(**kwargs, rng=np.random.default_rng(5))
    direct = MultipleNetworksSampler(
        **kwargs, rng=np.random.default_rng(5)
    ).sample_latent()
    result = network.generate()
    np.testing.assert_array_equal(result["Y"], direct["Y"])
    for actual, expected in zip(result["X"], direct["X"]):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("network_class", [GaussianNetwork, BernoulliNetwork])
def test_networks_forward_t3_errors_and_x_network_correlation(network_class):
    kwargs = dict(
        n=6,
        p=3,
        d_x=2,
        d_y=2,
        B=0,
        x_variance=2,
        x_network_correlation=0.5,
        eps_variance=3,
        eps_distribution="student_t_3",
    )
    network = network_class(**kwargs, rng=np.random.default_rng(51))
    direct = MultipleNetworksSampler(
        **kwargs, rng=np.random.default_rng(51)
    ).sample_latent()

    result = network.generate()

    assert network.latent_sampler.x_network_correlation == 0.5
    assert network.latent_sampler.eps_distribution == "student_t_3"
    np.testing.assert_array_equal(result["Y"], direct["Y"])
    for actual, expected in zip(result["X"], direct["X"]):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "edge_var", [-1, np.inf, -np.inf, np.nan, [1.0], np.eye(2), 1j]
)
def test_gaussian_network_rejects_invalid_edge_variance(edge_var):
    with pytest.raises(ValueError, match="edge_var"):
        GaussianNetwork(3, 2, 1, 1, edge_var=edge_var)


@pytest.mark.parametrize("parameter", ["n", "p", "d_x", "d_y"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_gaussian_network_validates_dimensions(parameter, value):
    kwargs = dict(n=3, p=2, d_x=1, d_y=1)
    kwargs[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        GaussianNetwork(**kwargs)


@pytest.mark.parametrize(
    "keyword",
    [
        "k",
        "ky",
        "Z",
        "Y",
        "X",
        "symmetric",
        "self_loops",
        "sparsity_exponent",
        "copula_model",
        "functional_form",
        "rdpg_distr",
        "latent_sim",
        "marginals",
    ],
)
def test_gaussian_network_rejects_removed_arguments(keyword):
    with pytest.raises(TypeError, match=keyword):
        GaussianNetwork(3, 2, 1, 1, **{keyword: None})


def test_gaussian_network_name_repr_and_default_rng():
    network = GaussianNetwork(4, 2, 3, 1)
    assert isinstance(network.rng, np.random.Generator)
    assert network.latent_sampler.rng is network.rng
    assert network.get_name() == "GaussianNetwork_multiple_networks"
    assert (
        repr(network)
        == "GaussianNetwork_multiple_networks(n=4, p=2, d_x=3, d_y=1, edge_var=1.0)"
    )


@pytest.mark.parametrize(
    "n,p,d_x,d_y", [(12, 3, 2, 4), (1, 1, 1, 1), (1, 3, 2, 1), (7, 2, 1, 3)]
)
@pytest.mark.parametrize("rdpg", [False, True])
def test_bernoulli_network_contract(n, p, d_x, d_y, rdpg):
    # B = 0 and bounded constant X give valid RDPG probabilities in all dimensions.
    result = BernoulliNetwork(
        n,
        p,
        d_x,
        d_y,
        rdpg=rdpg,
        B=np.zeros((d_y, p * d_x)),
        x_mean=0.1,
        x_variance=0,
        eps_variance=0,
        rng=np.random.default_rng(20),
    ).generate()
    assert set(result) == {"A_Y", "A_X", "Y", "X", "B"}
    assert isinstance(result["X"], list) and len(result["X"]) == p
    assert isinstance(result["A_X"], list) and len(result["A_X"]) == p
    assert result["Y"].shape == (n, d_y)
    assert result["B"].shape == (d_y, p * d_x)
    assert all(x.shape == (n, d_x) for x in result["X"])
    for adjacency in [result["A_Y"], *result["A_X"]]:
        assert adjacency.shape == (n, n)
        assert np.issubdtype(adjacency.dtype, np.integer)
        assert set(np.unique(adjacency)).issubset({0, 1})
        np.testing.assert_array_equal(adjacency, adjacency.T)
        np.testing.assert_array_equal(adjacency.diagonal(), np.zeros(n))


def test_bernoulli_rdpg_zero_and_one_probabilities_are_deterministic():
    result = BernoulliNetwork(
        5,
        2,
        1,
        1,
        rdpg=True,
        B=[[0.0, 0.0]],
        x_mean=[0.0, 1.0],
        x_variance=0,
        eps_variance=0,
        rng=np.random.default_rng(21),
    ).generate()
    np.testing.assert_array_equal(result["A_Y"], np.zeros((5, 5)))
    np.testing.assert_array_equal(result["A_X"][0], np.zeros((5, 5)))
    np.testing.assert_array_equal(result["A_X"][1], np.ones((5, 5)) - np.eye(5))


def test_bernoulli_preserves_linear_latents_and_block_order():
    B = np.array([[1.0, 2.0, 0.5, 1.0], [0.5, 0.0, 1.0, 0.0]])
    mean = np.array([0.1, 0.2, 0.3, 0.1])
    result = BernoulliNetwork(
        8,
        2,
        2,
        2,
        rdpg=True,
        B=B,
        x_mean=mean,
        x_variance=0,
        eps_variance=0,
        rng=np.random.default_rng(22),
    ).generate()
    np.testing.assert_array_equal(result["B"], B)
    for k, x in enumerate(result["X"]):
        np.testing.assert_array_equal(x, np.tile(mean[2 * k : 2 * (k + 1)], (8, 1)))
    np.testing.assert_array_equal(
        result["Y"], np.concatenate(result["X"], axis=1) @ B.T
    )


@pytest.mark.parametrize("rdpg", [False, True])
def test_bernoulli_reproducibility_across_repeated_generations(rdpg):
    kwargs = dict(n=10, p=2, d_x=2, d_y=3, rdpg=rdpg)
    if rdpg:
        kwargs.update(B=np.full((3, 4), 0.1), x_mean=0.2, x_variance=0, eps_variance=0)
    a = BernoulliNetwork(**kwargs, rng=np.random.default_rng(23))
    b = BernoulliNetwork(**kwargs, rng=np.random.default_rng(23))
    for _ in range(2):
        result_a, result_b = a.generate(), b.generate()
        for key in ("A_Y", "Y", "B"):
            np.testing.assert_array_equal(result_a[key], result_b[key])
        for key in ("A_X", "X"):
            for array_a, array_b in zip(result_a[key], result_b[key]):
                np.testing.assert_array_equal(array_a, array_b)


@pytest.mark.parametrize("rdpg", [0, 1, "false", "true", None, []])
def test_bernoulli_rejects_nonboolean_rdpg(rdpg):
    with pytest.raises(ValueError, match="rdpg.*bool"):
        BernoulliNetwork(3, 2, 1, 1, rdpg=rdpg)


@pytest.mark.parametrize("parameter", ["n", "p", "d_x", "d_y"])
def test_bernoulli_validates_dimensions(parameter):
    kwargs = dict(n=3, p=2, d_x=1, d_y=1)
    kwargs[parameter] = 0
    with pytest.raises(ValueError, match=parameter):
        BernoulliNetwork(**kwargs)


@pytest.mark.parametrize(
    "keyword",
    [
        "k",
        "ky",
        "Z",
        "Y",
        "X",
        "symmetric",
        "self_loops",
        "sparsity_exponent",
        "edge_var",
    ],
)
def test_bernoulli_rejects_obsolete_arguments(keyword):
    with pytest.raises(TypeError, match=keyword):
        BernoulliNetwork(3, 2, 1, 1, **{keyword: None})


@pytest.mark.parametrize("rdpg", [False, True])
def test_bernoulli_uses_exact_probabilities_and_shared_rng(rdpg):
    class RecordingGenerator(np.random.Generator):
        def __init__(self, seed):
            super().__init__(np.random.PCG64(seed))
            self.probabilities = []

        def binomial(self, n, p, size=None):
            self.probabilities.append(np.array(p, copy=True))
            return super().binomial(n, p, size=size)

    kwargs = dict(n=10, p=2, d_x=2, d_y=3)
    if rdpg:
        kwargs.update(B=np.full((3, 4), 0.1), x_mean=0.2, x_variance=0, eps_variance=0)
    else:
        kwargs.update(
            x_mean=[-1.0, 2.0, 0.0, 1.0],
            x_variance=np.eye(4) + 0.1 * np.ones((4, 4)),
            eps_variance=np.eye(3) + 0.2 * np.ones((3, 3)),
            b_distribution="multivariate_gaussian",
            b_mean=-0.5,
            b_variance=2,
        )
    rng, reference_rng = RecordingGenerator(24), np.random.default_rng(24)
    network = BernoulliNetwork(**kwargs, rdpg=rdpg, rng=rng)
    assert network.rng is rng
    assert network.latent_sampler.rng is rng
    result = network.generate()
    reference = MultipleNetworksSampler(**kwargs, rng=reference_rng).sample_latent()

    # The link function must change only edge probabilities, never latent outputs.
    np.testing.assert_array_equal(result["B"], reference["B"])
    for actual, expected in zip(
        [result["Y"], *result["X"]], [reference["Y"], *reference["X"]]
    ):
        np.testing.assert_array_equal(actual, expected)
    assert len(rng.probabilities) == kwargs["p"] + 1
    for adjacency, latent, passed_probabilities in zip(
        [result["A_Y"], *result["A_X"]],
        [reference["Y"], *reference["X"]],
        rng.probabilities,
    ):
        probabilities = latent @ latent.T
        if not rdpg:
            probabilities = expit(probabilities)
        np.fill_diagonal(probabilities, 0)
        np.testing.assert_array_equal(passed_probabilities, probabilities)
        sampled = reference_rng.binomial(1, probabilities)
        upper = np.triu(sampled, k=1)
        np.testing.assert_array_equal(adjacency, upper + upper.T)
    assert rng.bit_generator.state == reference_rng.bit_generator.state


@pytest.mark.parametrize(
    "latent",
    [
        [[1.0], [-1.0]],
        [[2.0], [2.0]],
        [[np.nan], [1.0]],
        [[np.inf], [1.0]],
    ],
)
def test_bernoulli_rdpg_rejects_invalid_probabilities_without_clipping(latent):
    network = BernoulliNetwork(2, 1, 1, 1, rdpg=True)
    with pytest.raises(ValueError, match=r"rdpg=True.*\[0, 1\]"):
        network._sample_adjacency(np.array(latent))


def test_bernoulli_rdpg_ignores_unused_self_loop_probabilities():
    network = BernoulliNetwork(2, 1, 2, 1, rdpg=True)
    # Norms exceed one, but the only edge probability is 0.5.
    latent = np.array([[2.0, 0.0], [0.25, 2.0]])
    adjacency = network._sample_adjacency(latent)
    np.testing.assert_array_equal(adjacency, adjacency.T)
    np.testing.assert_array_equal(adjacency.diagonal(), [0, 0])


def test_bernoulli_logistic_handles_large_positive_and_negative_inner_products():
    network = BernoulliNetwork(3, 1, 1, 1)
    latent = np.array([[1000.0], [-1000.0], [1000.0]])
    adjacency = network._sample_adjacency(latent)
    np.testing.assert_array_equal(adjacency, [[0, 0, 1], [0, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize("rdpg", [False, np.bool_(True)])
def test_bernoulli_name_repr_and_default_generator(rdpg):
    network = BernoulliNetwork(4, 2, 3, 1, rdpg=rdpg)
    assert network.get_name() == "BernoulliNetwork_multiple_networks"
    assert repr(network) == (
        f"BernoulliNetwork_multiple_networks(n=4, p=2, d_x=3, d_y=1, rdpg={bool(rdpg)})"
    )
    assert isinstance(network.rng, np.random.Generator)
    assert network.latent_sampler.rng is network.rng
