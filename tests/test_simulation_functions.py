import numpy as np

from src.dgp import GaussianNetwork
from src.helper_functions.simulation_functions import run_scenario
from src.methods import RVTest
from src.metrics import ComputeAll, Rejection, ReturnMetric
from src.solvers.weighted_network import ASE


def test_run_scenario_does_not_require_single_adjacency_or_return_density():
    # Isolate density handling from the runner's separate argument migration:
    # its factory still receives both DGP and simulation/method arguments.
    def dgp_factory(**kwargs):
        return GaussianNetwork(
            n=kwargs["n"],
            p=kwargs["p"],
            d_x=kwargs["d_x"],
            d_y=kwargs["d_y"],
            rng=kwargs["rng"],
        )

    args = {
        "setup": (dgp_factory, ASE),
        "method": RVTest,
        "n": 10,
        "p": 2,
        "d_x": 1,
        "d_y": 2,
        "npermutations": 2,
    }
    result = run_scenario(
        [ComputeAll(), Rejection(), ReturnMetric()],
        args,
        seed=np.random.SeedSequence(1),
    )
    assert "density" not in result
    assert result["args"]["dgp_name"] == "GaussianNetwork_multiple_networks"
    assert result["ReturnMetric"]["Y"].shape == (10, 2)
    assert result["ReturnMetric"]["X"].shape == (10, 2)
    assert result["Rejection"] == result["ComputeAll"]["Rejection"]
    assert np.isnan(result["ComputeAll"]["FalseRejection"])
    for prefix in ("RelativeFrobeniusNorm", "ProcrustesDistance"):
        for suffix in ("Y", "X_1", "X_2", "X_global"):
            assert np.isfinite(result["ComputeAll"][f"{prefix}_{suffix}"])
