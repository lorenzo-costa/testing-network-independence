"""Small power/type-I-error study for the scalar functional alternatives.

Run from the repository root with no arguments:

    python -m src.run_temp_sim

Edit the constants in the first section to change the study or test method.
"""

from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd

from src.dgp import BernoulliNetwork
from src.helper_functions.simulation_functions import run_simulation
from src.methods import DistanceCorrelationTest, MultivariateACTest
from src.metrics import ReturnMetric
from src.solvers.weighted_network import ASE


# ---------------------------------------------------------------------------
# Self-contained study specification
# ---------------------------------------------------------------------------
N = 25
K = 1
MC_RUNS = 200
N_PERMUTATIONS = 100
ALPHA = 0.05
SEED = 1
N_JOBS = None  # None asks run_simulation to use all available CPU cores.

SNRS = [0.5, 0.3]
PREDICTOR_DISTRIBUTIONS = ["gaussian"]

# Add or replace entries with any repository test implementing the same API.
# Keep each test's n_jobs at one because run_simulation parallelizes the Monte
# Carlo scenarios.
TEST_METHODS = (
    ("AC (adaptive)", partial(MultivariateACTest, adaptive_m=True, n_jobs=1)),
    ("MGC", partial(DistanceCorrelationTest, test_method="mgc", n_jobs=1)),
)

# These are the k=1 functionals and distribution-specific overrides from
# config_functionals.yaml.
FUNCTIONALS = (
    # Monotonic alternatives (disabled):
    # ("linear", {"weights": [0.5], "normalize_index": True}, {}),
    (
        "piecewise",
        {
            "weights": [0.5],
            "normalize_index": True,
            "switch_feature": 0,
            "threshold": 0.5,
        },
        {},
    ),
    # (
    #     "threshold",
    #     {
    #         "weights": [0.5],
    #         "normalize_index": True,
    #         "threshold": 0.5,
    #         "global_scale": 10.0,
    #     },
    #     {},
    # ),
    (
        "sigmoid",
        {
            "weights": [0.5],
            "normalize_index": True,
            "threshold": 0.0,
            "slope": 5.0,
            "global_scale": 10.0,
        },
        {"uniform_rdpg": {"threshold": 0.5, "slope": 17.0}},
    ),
    (
        "sine",
        {
            "weights": [0.5],
            "normalize_index": True,
            "frequency": 3.0,
            "global_scale": 10.0,
        },
        {"uniform_rdpg": {"frequency": 15.0}},
    ),
    (
        "abs_max",
        {"center": 0.0, "input_scale": 1.0},
        {"uniform_rdpg": {"center": 0.5, "input_scale": 6.0}},
    ),
    (
        "tanh_product",
        {"input_scale": 1.0, "global_scale": 10.0, "center": 0.0},
        {"uniform_rdpg": {"input_scale": 6.0, "center": 0.5}},
    ),
)

MONOTONIC_FUNCTIONALS = frozenset(
    {"linear", "threshold", "sigmoid", "tanh_product"}
)


class PermutedResponseBernoulliNetwork(BernoulliNetwork):
    """Generate the configured signal, then permute it to impose the null."""

    def generate(self):
        data = super().generate()
        data["Y"] = data["Y"][self.rng.permutation(self.n)]
        self.is_null = True
        return data


def build_design():
    """Return matched alternative/null rows for run_simulation."""
    design = []
    for distribution in PREDICTOR_DISTRIBUTIONS:
        for functional, base_params, overrides in FUNCTIONALS:
            params = deepcopy(base_params)
            params.update(overrides.get(distribution, {}))
            for snr in SNRS:
                for test_label, test_method in TEST_METHODS:
                    common = {
                        "method": test_method,
                        "test_label": test_label,
                        "n": N,
                        "k": K,
                        "ky": K,
                        "alpha": ALPHA,
                        "rho": snr,
                        "noise_scale": 0.0,
                        "noise_type": "additive",
                        "edge_var": 3,
                        "npermutations": N_PERMUTATIONS,
                        "use_true_latent": True,
                        "rdpg": "auto",
                        "predictor_distribution": distribution,
                        "functional_form": functional,
                        "function_params": params,
                    }
                    design.append(
                        {
                            **common,
                            "setup": (BernoulliNetwork, ASE),
                            "hypothesis": "alternative",
                        }
                    )
                    design.append(
                        {
                            **common,
                            "setup": (PermutedResponseBernoulliNetwork, ASE),
                            "hypothesis": "null",
                        }
                    )
    return design


def summarize(results):
    """Put both tests side by side for each matched DGP setting."""
    records = []
    for result in results:
        args = result["args"]
        records.append(
            {
                "method": args["test_label"],
                "predictor": args["predictor_distribution"],
                "functional": args["functional_form"],
                "snr": args["rho"],
                "hypothesis": args["hypothesis"],
                "rejected": result["ReturnMetric"] < ALPHA,
            }
        )

    rates = (
        pd.DataFrame.from_records(records)
        .groupby(
            ["method", "predictor", "functional", "snr", "hypothesis"],
            sort=False,
        )["rejected"]
        .mean()
        .unstack("hypothesis")
        .rename(columns={"alternative": "power", "null": "type_I_error"})
        .reset_index()
    )
    table = rates.pivot(
        index=["predictor", "functional", "snr"],
        columns="method",
        values=["power", "type_I_error"],
    )
    table.columns = [f"{method}_{metric}" for metric, method in table.columns]
    table = table.rename(
        columns={
            "MGC_power": "MGC power",
            "MGC_type_I_error": "MGC type I",
            "AC (adaptive)_power": "AC power",
            "AC (adaptive)_type_I_error": "AC type I",
        }
    ).reset_index()
    return table[
        [
            "predictor",
            "functional",
            "snr",
            "MGC power",
            "MGC type I",
            "AC power",
            "AC type I",
        ]
    ]


def main():
    design = build_design()
    total = MC_RUNS * len(design)
    print(
        f"Running {total} scenarios ({MC_RUNS} MC runs x {len(design)} rows) "
        f"with n={N}, k={K}, {N_PERMUTATIONS} test permutations."
    )
    results = run_simulation(
        nsim=MC_RUNS,
        factorial_design=design,
        metrics=[ReturnMetric(only_return="p-value")],
        parallel=True,
        rng=np.random.default_rng(SEED),
        n_jobs=N_JOBS,
    )
    table = summarize(results)
    formatters = {
        "snr": "{:.1f}".format,
        "MGC power": "{:.3f}".format,
        "MGC type I": "{:.3f}".format,
        "AC power": "{:.3f}".format,
        "AC type I": "{:.3f}".format,
    }
    monotonic = table[table["functional"].isin(MONOTONIC_FUNCTIONALS)]
    non_monotonic = table[~table["functional"].isin(MONOTONIC_FUNCTIONALS)]

    print(f"\nMonotonic alternatives (nominal alpha = {ALPHA:.2f})")
    if monotonic.empty:
        print("No monotonic functionals are active.")
    else:
        print(monotonic.to_string(index=False, formatters=formatters))
    print(f"\nNon-monotonic alternatives (nominal alpha = {ALPHA:.2f})")
    print(
        non_monotonic.to_string(
            index=False,
            formatters=formatters,
        )
    )
    return monotonic, non_monotonic


if __name__ == "__main__":
    main()

