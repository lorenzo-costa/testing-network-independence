import numpy as np
import pytest

from src.dgp import GaussianNetwork
from src.helper_functions import simulation_functions
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


def test_parallel_simulation_configures_one_blas_thread_per_worker(monkeypatch):
    captured = {}

    class FakePool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap_unordered(self, function, tasks, chunksize):
            return []

    monkeypatch.setattr(simulation_functions, "Pool", FakePool)
    simulation_functions.run_simulation_parallel(
        nsim=0,
        factorial_design=[],
        metrics=[],
        n_jobs=2,
        blas_threads=1,
    )

    assert captured["initializer"] is simulation_functions._initialize_parallel_worker
    assert captured["initargs"] == (1,)


@pytest.mark.parametrize("blas_threads", [0, -1, 1.5, True, None])
def test_parallel_simulation_rejects_invalid_blas_thread_limits(blas_threads):
    with pytest.raises(ValueError, match="blas_threads"):
        simulation_functions.run_simulation_parallel(
            nsim=0,
            factorial_design=[],
            metrics=[],
            n_jobs=1,
            blas_threads=blas_threads,
        )


def test_run_simulation_forwards_blas_thread_limit(monkeypatch):
    captured = {}

    def capture_parallel(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        simulation_functions, "run_simulation_parallel", capture_parallel
    )
    simulation_functions.run_simulation(
        nsim=0,
        factorial_design=[],
        metrics=[],
        parallel=True,
        blas_threads=3,
    )
    assert captured["blas_threads"] == 3
