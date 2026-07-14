import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.methods import (
    CanonicalCorrelationTest,
    DistanceCorrelationTest,
    LLKRatioTest,
    MultivariateACTest,
    ObservedCVM,
    QAP,
    RVTest,
)
from src.solvers.weighted_network import ASE
from src.test_functions.cvm_statistic import observed_cvm_dependency


NETWORK_MODELS = [
    pytest.param(GaussianNetwork, id="gaussian-network"),
    pytest.param(BernoulliNetwork, id="bernoulli-network"),
]


LATENT_SAMPLERS = [
    pytest.param(
        {"copula_model": "gaussian", "marginals": "gaussian", "rho": 0.25},
        id="gaussian-copula",
    ),
    pytest.param(
        {"rdpg_distr": "dirichlet", "rho": 0.25},
        id="dirichlet-rdpg",
    ),
    pytest.param(
        {"latent_sim": "linear"},
        id="hyppo-linear",
    ),
    pytest.param(
        {"dim_common": 1, "shared_latent_type": "gaussian"},
        id="orthogonal-gaussian",
    ),
    pytest.param(
        {"functional_form": "linear", "kx": 1},
        id="functional-linear",
    ),
    pytest.param(
        {
            "sbm_covariate_sampling": "step",
            "x_distribution": "gaussian",
        },
        id="sbm-continuous-covariate",
    ),
]


METHOD_NAMES = [
    pytest.param("rv", id="rv-test"),
    pytest.param("canonical-correlation", id="canonical-correlation-test"),
    pytest.param("multivariate-ac", id="multivariate-ac-test"),
    pytest.param("observed-cvm", id="observed-cvm-test"),
    pytest.param("distance-correlation", id="distance-correlation-test"),
    pytest.param("likelihood-ratio", id="likelihood-ratio-test"),
    pytest.param("qap", id="qap"),
]


def build_method(method_name, rng):
    common = {
        "k": 2,
        "solver": ASE,
        "rng": rng,
    }

    if method_name == "rv":
        return RVTest(npermutations=1, **common)
    if method_name == "canonical-correlation":
        return CanonicalCorrelationTest(npermutations=1, **common)
    if method_name == "multivariate-ac":
        return MultivariateACTest(M=1, npermutations=1, **common)
    if method_name == "observed-cvm":
        return ObservedCVM(
            test_function=observed_cvm_dependency,
            npermutations=1,
            rng=rng,
        )
    if method_name == "distance-correlation":
        return DistanceCorrelationTest(npermutations=1, **common)
    if method_name == "likelihood-ratio":
        return LLKRatioTest(approximation="chi-sq", **common)
    if method_name == "qap":
        return QAP(npermutations=1, rng=rng)
    raise ValueError(f"Unknown integration-test method: {method_name}")


def prepare_method_data(method_name, generated, rng):
    if method_name != "likelihood-ratio":
        return generated

    # LLKRatioTest's raw-adjacency branch currently references a local variable
    # before assignment. Supplying the estimates its public API accepts keeps
    # this integration test on the working path; the mismatch is documented in
    # UNEXPECTED_BEHAVIOURS.txt.
    data = dict(generated)
    data["estimated_Z"] = ASE(generated["A"], k=2, rng=rng)[0]
    data["estimated_X"] = ASE(generated["B"], k=2, rng=rng)[0]
    return data


@pytest.mark.parametrize("network_type", NETWORK_MODELS)
@pytest.mark.parametrize("latent_kwargs", LATENT_SAMPLERS)
@pytest.mark.parametrize("method_name", METHOD_NAMES)
def test_network_sampling_and_independence_method_pipeline_runs(
    network_type,
    latent_kwargs,
    method_name,
):
    network_kwargs = dict(latent_kwargs)
    if network_type is GaussianNetwork:
        network_kwargs["edge_var"] = 0.25
    elif network_kwargs.get("rdpg_distr") is not None:
        network_kwargs["rdpg"] = True

    network = network_type(
        n=14,
        k=2,
        symmetric=True,
        self_loops=False,
        rng=np.random.default_rng(101),
        **network_kwargs,
    )
    generated = network.generate()

    method_rng = np.random.default_rng(202)
    method = build_method(method_name, method_rng)
    method_data = prepare_method_data(method_name, generated, method_rng)
    method.fit(method_data)

    assert hasattr(method, "pvalue")
    assert hasattr(method, "reject_null")
    assert isinstance(method.reject_null, bool)

