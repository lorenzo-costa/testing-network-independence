"""Compare three AC tests with MGC under a conditional Gaussian copula.

The Bernoulli RDPG, stratum-specific marginals, and Model-X tests are
specified below. By default, class 0 uses rho and class 1 uses -rho.
At rho=0 the rejection rate estimates Type I error.

Example:
    python power_temp.py --nsim 10 --n 50 100 --rho 0 0.3 0.5
"""

import argparse
from functools import partial
from itertools import product

import numpy as np

from src.dgp import BernoulliNetwork
from src.helper_functions.simulation_functions import run_simulation
from src.methods.ac_test import MultivariateACTest
from src.methods.distance_correlation_test import DistanceCorrelationTest
from src.methods.cca_test import CanonicalCorrelationTest
from src.metrics import ReturnMetric
from src.solvers.weighted_network import ASE

# Edit these defaults to run the experiment without command-line options.
DEFAULT_NSIM = 10
DEFAULT_SAMPLE_SIZES = (200)
DEFAULT_RHOS = (0.5)
CLASS_RHO_SIGNS = (1, -1)
DEFAULT_NPERMUTATIONS = 100
DEFAULT_JOBS = 4
MIN_SAMPLE_SIZE = 20

SEED = 1
ALPHA = 0.05
LATENT_DIMENSION = 3
RESPONSE_DIMENSION = 3
USE_TRUE_LATENT = False
EMBEDDING_SOLVER = ASE

NETWORK_OPTIONS = {
    "rdpg": True,
    "conditional_copula": "gaussian",
    "C": 2,
    "class_probabilities": (0.5, 0.5),
    "center_latent": False,
}

# Each pair lists the coordinate marginals for X=0 and X=1, in that order.
MARGINAL_SETUPS = (
    (
        {"z": "uniform", "y": "gaussian"},
        {"z": "uniform", "y": "gaussian"},
    ),
)

MODEL_X_OPTIONS = {"test_type": "model_X", "conditional_generator": "from_dgp"}
AC_OPTIONS = {**MODEL_X_OPTIONS, "use_permutation_coeff": True}
METHODS = (
    ("AC (1 neighbor)", partial(MultivariateACTest, M=1, **AC_OPTIONS)),
    ("AC (sqrt(n))", partial(MultivariateACTest, M=0.5, **AC_OPTIONS)),
    ("AC adaptive", partial(MultivariateACTest, adaptive_m=True, **AC_OPTIONS)),
    (
        "MGC",
        partial(
            DistanceCorrelationTest,
            test_method="mgc",
            permutation_type="covariate",
            **MODEL_X_OPTIONS,
        ),
    ),
    (
        "CCA", partial(
            CanonicalCorrelationTest, 
            permutation_type="covariate", 
            **MODEL_X_OPTIONS)
    )
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--nsim", type=int, default=DEFAULT_NSIM,
        help="Replicates per setup, sample size, and rho.",
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES,
        help=f"One or more sample sizes (minimum {MIN_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--rho", type=float, nargs="+", default=DEFAULT_RHOS,
        help="One or more base copula correlations in [-1, 1], before class signs.",
    )
    parser.add_argument(
        "--npermutations", type=int, default=DEFAULT_NPERMUTATIONS,
        help="Model-X null draws per test.",
    )
    parser.add_argument(
        "--jobs", type=int, default=DEFAULT_JOBS,
        help="Number of simulation workers.",
    )
    args = parser.parse_args()

    if args.nsim < 1 or args.npermutations < 1 or args.jobs < 1:
        parser.error("--nsim, --npermutations, and --jobs must be positive.")
    # Tiny samples can yield constant adaptive AC components that the test
    # cannot standardize.
    # if not args.n or any(n < MIN_SAMPLE_SIZE for n in args.n):
    #     parser.error(f"--n must contain sample sizes of at least {MIN_SAMPLE_SIZE}.")
    # if not args.rho or any(not -1 <= rho <= 1 for rho in args.rho):
    #     parser.error("--rho must contain finite correlations in [-1, 1].")

    if isinstance(args.n, int):
        args.n = [args.n]
    else:
        args.n = list(dict.fromkeys(args.n))

    if isinstance(args.rho, float):
        args.rho = [args.rho]
    else:
        args.rho = list(dict.fromkeys(args.rho))
        
    return args


def experiment_design(args):
    design = []
    for setup_id, marginals in enumerate(MARGINAL_SETUPS, start=1):
        dgp = partial(BernoulliNetwork, marginals_by_class=marginals, **NETWORK_OPTIONS)
        for n, rho in product(args.n, args.rho):
            generation = {
                "setup": (dgp, EMBEDDING_SOLVER),
                "n": n,
                "rho": tuple(rho * sign for sign in CLASS_RHO_SIGNS),
                "k": LATENT_DIMENSION,
                "ky": RESPONSE_DIMENSION,
            }
            for label, method in METHODS:
                design.append({
                    **generation,
                    "_generation_keys": tuple(generation),
                    "setup_id": setup_id,
                    "rho_strength": rho,
                    "method_label": label,
                    "method": method,
                    "alpha": ALPHA,
                    "npermutations": args.npermutations,
                    "use_true_latent": USE_TRUE_LATENT,
                })
    return design


def result_key(row):
    return row["setup_id"], row["n"], row["rho_strength"], row["method_label"]


def rejection_rates(design, results, nsim):
    pvalues = {result_key(row): [] for row in design}
    for result in results:
        key = result_key(result["args"])
        pvalue = result["ReturnMetric"]
        if pvalue is None or not 0 <= pvalue <= 1:
            raise RuntimeError(f"Invalid p-value for {key}: {pvalue}")
        pvalues[key].append(pvalue)

    rates = {}
    for row in design:
        key = result_key(row)
        values = pvalues[key]
        if len(values) != nsim:
            raise RuntimeError(f"Expected {nsim} p-values for {key}, got {len(values)}")
        rates[key] = np.mean(np.asarray(values) < row["alpha"])
    return rates


def print_results(rates, args):
    methods = [label for label, _ in METHODS]
    signs = ", ".join(f"X={i} {sign:+g}" for i, sign in enumerate(CLASS_RHO_SIGNS))
    print(f"alpha={ALPHA:g}, nsim={args.nsim}, null draws={args.npermutations}")
    print(f"Class multipliers for rho ({signs})")
    print("Rejection rates estimate power, or Type I error when rho=0.")
    for setup_id, setup_marginals in enumerate(MARGINAL_SETUPS, start=1):
        print(f"\nSetup {setup_id}")
        for stratum, marginals in enumerate(setup_marginals):
            print(f"  X={stratum}  Z: {marginals['z']}  Y: {marginals['y']}")

        print(f"{'n':>6}  {'rho':>6}  " + "  ".join(f"{m:>16}" for m in methods))
        for n in args.n:
            for rho in args.rho:
                values = [rates[setup_id, n, rho, method] for method in methods]
                print(f"{n:6d}  {rho:6g}  " + "  ".join(f"{v:16.3f}" for v in values))


def main():
    args = parse_args()
    design = experiment_design(args)
    results = run_simulation(
        nsim=args.nsim,
        factorial_design=design,
        metrics=[ReturnMetric(only_return="p-value")],
        rng=np.random.default_rng(SEED),
        parallel=args.jobs > 1,
        n_jobs=args.jobs,
    )
    print_results(rejection_rates(design, results, args.nsim), args)


if __name__ == "__main__":
    main()
