"""Compare AC and distance-correlation power across sine frequencies.

The default study uses a Bernoulli network with Gaussian predictors, true
latent positions, n=100, and SNR values 0.1, 0.3, and 0.5. It compares
asymptotic AC with k=1, asymptotic AC with k=sqrt(n), adaptive permutation AC,
and the DC/MGC permutation test.

Examples
--------
Run the default study::

    python inspect_ac_type1_grid.py --nsim 100 --jobs 4

Run a quick smoke study at one SNR::

    python inspect_ac_type1_grid.py --nsim 10 --snrs 0.7 --jobs 1

Study the higher frequencies used for uniform-RDPG predictors::

    python inspect_ac_type1_grid.py \
        --predictor-distribution uniform_rdpg \
        --frequencies 5 10 15 30
"""

import argparse
from functools import partial
from itertools import product

import numpy as np

from src.dgp import BernoulliNetwork
from src.helper_functions.simulation_functions import run_simulation
from src.methods.ac_test import MultivariateACTest
from src.methods.distance_correlation_test import DistanceCorrelationTest
from src.metrics import ReturnMetric
from src.solvers.MaMa_uuuuu import pgd_fit_wrapper


DEFAULT_FREQUENCIES = (3.0, 3.5)
DEFAULT_SNRS = (0.1, 0.3, 0.5)
ALPHA = 0.05
METHODS = (
    "AC asymptotic (k=1)",
    "AC asymptotic (k=sqrt(n))",
    "AC adaptive (permutation)",
    "DC/MGC",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsim", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--npermutations", type=int, default=100)
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs="+",
        default=DEFAULT_FREQUENCIES,
        help="Sine frequencies to compare (default: 3 3.5).",
    )
    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=DEFAULT_SNRS,
        help="Signal fractions to compare (default: 0.1 0.3 0.5).",
    )
    parser.add_argument(
        "--predictor-distribution",
        choices=("gaussian", "uniform_rdpg"),
        default="gaussian",
    )
    latent_group = parser.add_mutually_exclusive_group()
    latent_group.add_argument(
        "--use-true-latent",
        dest="use_true_latent",
        action="store_true",
        help="Use true latent positions (the default).",
    )
    latent_group.add_argument(
        "--use-estimated-latent",
        dest="use_true_latent",
        action="store_false",
        help="Estimate latent positions from the observed network.",
    )
    parser.set_defaults(use_true_latent=True)
    args = parser.parse_args()

    if args.nsim < 1 or args.jobs < 1 or args.n < 2:
        parser.error("nsim and jobs must be positive, and n must be at least 2")
    if args.npermutations < 1:
        parser.error("npermutations must be positive")
    if any(not np.isfinite(value) or value <= 0 for value in args.frequencies):
        parser.error("frequencies must contain positive finite values")
    if any(not np.isfinite(value) or not 0 < value <= 1 for value in args.snrs):
        parser.error("snrs must contain finite values in (0, 1]")
    return args


def sine_parameters(frequency):
    return {
        "weights": [0.5, 0.5, 0.5, 0.5],
        "normalize_index": True,
        "frequency": frequency,
        "global_scale": 10.0,
    }


def method_designs(args):
    return (
        (
            "AC asymptotic (k=1)",
            partial(
                MultivariateACTest,
                M=1,
                test_type="asymptotic",
                permutation_type="covariate",
                use_permutation_coeff=False,
            ),
        ),
        (
            "AC asymptotic (k=sqrt(n))",
            partial(
                MultivariateACTest,
                M=0.5,
                test_type="asymptotic",
                permutation_type="covariate",
                use_permutation_coeff=False,
            ),
        ),
        (
            "AC adaptive (permutation)",
            partial(
                MultivariateACTest,
                adaptive_m=True,
                test_type="permutation",
                permutation_type="covariate",
                use_permutation_coeff=False,
                npermutations=args.npermutations,
            ),
        ),
        (
            "DC/MGC",
            partial(
                DistanceCorrelationTest,
                test_method="mgc",
                npermutations=args.npermutations,
                permutation_type="covariate",
            ),
        ),
    )


def build_factorial_design(args):
    design = []
    for frequency, snr in product(args.frequencies, args.snrs):
        setup = (
            partial(
                BernoulliNetwork,
                k=4,
                ky=1,
                rdpg="auto",
                predictor_distribution=args.predictor_distribution,
                center_latent=False,
                functional_form="sine",
                function_params=sine_parameters(frequency),
            ),
            pgd_fit_wrapper,
        )
        for method_label, method in method_designs(args):
            design.append(
                {
                    "setup": setup,
                    "method": method,
                    "method_label": method_label,
                    "frequency": frequency,
                    "n": args.n,
                    # FunctionalGenerator uses rho as the target signal fraction.
                    "rho": snr,
                    "alpha": ALPHA,
                    "use_true_latent": args.use_true_latent,
                    # Frequency is already captured by the setup partial. Keeping
                    # it outside the generation keys makes it available for output
                    # without passing it twice to the DGP.
                    "_generation_keys": ("setup", "n", "rho"),
                }
            )
    return design


def rejection_rates(results, args):
    pvalues = {
        (frequency, snr, method): []
        for frequency, snr, method in product(
            args.frequencies, args.snrs, METHODS
        )
    }
    for result in results:
        row = result["args"]
        key = (row["frequency"], row["rho"], row["method_label"])
        pvalues[key].append(result["ReturnMetric"])

    rates = {}
    for key, values in pvalues.items():
        if len(values) != args.nsim:
            raise RuntimeError(
                f"Expected {args.nsim} results for {key}, found {len(values)}"
            )
        rates[key] = np.count_nonzero(np.asarray(values) < ALPHA) / args.nsim
    return rates


def print_results(rates, args):
    latent_label = "true" if args.use_true_latent else "estimated"
    print(
        f"alpha={ALPHA}, n={args.n}, nsim={args.nsim}, "
        f"predictors={args.predictor_distribution}, latent={latent_label}, "
        f"permutations={args.npermutations}"
    )
    header = (
        f"{'frequency':>9}  {'SNR':>4}  "
        + "  ".join(f"{method:>27}" for method in METHODS)
    )
    print(header)
    for frequency, snr in product(args.frequencies, args.snrs):
        values = [rates[(frequency, snr, method)] for method in METHODS]
        formatted_values = "  ".join(f"{value:27.3f}" for value in values)
        print(f"{frequency:9g}  {snr:4.2f}  {formatted_values}")
    print("Values are empirical rejection rates (power) at alpha=0.05.")


def main():
    args = parse_args()
    results = run_simulation(
        nsim=args.nsim,
        metrics=[ReturnMetric(only_return="p-value")],
        factorial_design=build_factorial_design(args),
        rng=np.random.default_rng(2),
        parallel=args.jobs > 1,
        n_jobs=args.jobs,
    )
    print_results(rejection_rates(results, args), args)


if __name__ == "__main__":
    main()
