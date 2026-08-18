import numpy as np

from src.dgp import GaussianNetwork
from src.helper_functions.simulation_functions import run_scenario
from src.methods import RVTest
from src.metrics import ComputeAll, ReturnMetric
from src.solvers.weighted_network import ASE


def test_run_scenario_returns_all_linear_model_latent_blocks_and_metrics():
    args = {
        "setup": (GaussianNetwork, ASE),
        "method": RVTest,
        "n": 10,
        "k": [1, 2, 3],
        "B": "zero",
        "alpha": 0.05,
        "edge_var": 1,
        "marginals": "gaussian",
        "npermutations": 2,
    }
    result = run_scenario(
        [ReturnMetric(), ComputeAll()],
        args,
        seed=np.random.SeedSequence(1),
    )

    assert result["ReturnMetric"]["Y"].shape == (10, 1)
    assert [block.shape for block in result["ReturnMetric"]["X"]] == [
        (10, 2),
        (10, 3),
    ]
    assert len(result["ComputeAll"]["RelativeFrobeniusNorm"]) == 3
    assert len(result["ComputeAll"]["ProcrustesDistance"]) == 3
