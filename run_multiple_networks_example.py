"""Compare RV, CCA, and MGC across network models and latent SNR settings."""

from functools import partial
from itertools import product

import numpy as np
import pandas as pd

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.helper_functions.multiple_network_factories import make_network
from src.helper_functions.simulation_functions import run_simulation
from src.methods import (
    CanonicalCorrelationTest,
    DistanceCorrelationTest,
    RVTest,
)
from src.metrics import ComputeAll, ReturnMetric
from src.solvers.MaMa_uuuuu import pgd_fit_wrapper
from src.solvers.weighted_network import ASE


def run_experiment(
    nsim=100,
    n=200,
    npermutations=200,
    n_jobs=-1,
    *,
    snr_values=(0, 0.01, 0.05, 0.1),
):
    """Run repetitions for every network, method, and SNR combination.

    SNR zero uses B=0; positive SNR values use newly sampled, calibrated B.
    With the defaults, 2 networks x 3 methods x 5 SNRs x 50 repetitions
    produces 1,500 simulation runs.
    """
    snr_values = tuple(snr_values)
    if not snr_values:
        raise ValueError("snr_values must contain at least one setting.")
    setup = [
        (partial(make_network, GaussianNetwork), ASE),
        # Explicit NumPy backend avoids starting JAX runtimes in every worker.
        (
            partial(make_network, BernoulliNetwork),
            partial(pgd_fit_wrapper),
        ),
    ]
    methods = [
        (RVTest, "RV"),
        (CanonicalCorrelationTest, "CCA"),
        (partial(DistanceCorrelationTest, test_method="mgc"), "MGC"),
    ]
    factorial_design = [
        dict(
            setup=pair,
            method=method,
            method_label=method_label,
            n=n,
            p=5,
            d_x=5,
            d_y=5,
            B=0 if snr == 0 else None,
            snr=snr,
            hypothesis="H0" if snr == 0 else "H1",
            alpha=0.05,
            edge_var=1,
            use_true_latent=False,
            approximation="permutation",
            permutation_type="latent",
            npermutations=npermutations,
            n_jobs=1,  # Parallelize simulations, not permutations inside workers.
        )
        for pair, (method, method_label), snr in product(setup, methods, snr_values)
    ]
    out = run_simulation(
        nsim=nsim,
        metrics=[ComputeAll(), ReturnMetric("p-value")],
        factorial_design=factorial_design,
        rng=np.random.default_rng(2),
        parallel=True,
        n_jobs=None if n_jobs == -1 else n_jobs,  # The runner uses None for all CPUs.
        batch_size=16,
    )
    return pd.DataFrame(
        [
            {
                "network": row["args"]["dgp_name"],
                "method": row["args"]["method_label"],
                "hypothesis": row["args"]["hypothesis"],
                "snr": row["args"]["snr"],
                "p-value": row["ReturnMetric"],
                **row["ComputeAll"],
            }
            for row in out
        ]
    )


if __name__ == "__main__":  # Required for multiprocessing on macOS/Windows.
    results = run_experiment()
    results.to_csv("multiple_network_results.csv", index=False)
    print(
        results.groupby(["network", "method", "snr", "hypothesis"])["Rejection"].mean()
    )
