import numpy as np

from src.dgp import GaussianNetwork
from src.helper_functions.simulation_functions import run_scenario
from src.methods import RVTest
from src.metrics import ReturnMetric
from src.solvers.weighted_network import ASE


def test_run_scenario_uses_single_network_density_and_returns_y():
    args = {
        "setup": (GaussianNetwork, ASE),
        "method": RVTest,
        "n": 10,
        "k": 2,
        "ky": 1,
        "rho": 0.2,
        "alpha": 0.05,
        "edge_var": 1,
        "marginals": "gaussian",
        "copula_model": "gaussian",
        "npermutations": 2,
    }
    result = run_scenario([ReturnMetric()], args, seed=np.random.SeedSequence(1))
    assert isinstance(result["density"], float)
    assert result["ReturnMetric"]["Y"].shape == (10, 1)
