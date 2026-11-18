"""Run a small functional-power experiment for the legacy Bernoulli setup.

The experiment uses Gaussian latent predictors, a logistic Bernoulli network,
PGD-estimated latent positions, and additive response noise with fixed standard
deviation 2.  No YAML configuration is loaded.

The five tests are DC/MGC, permutation CCA, asymptotic AC with M=1,
asymptotic AC with M=sqrt(n), and adaptive permutation AC.

Examples
--------
Run the default small experiment::

    python power_experiment.py

Run a quick smoke experiment::

    python power_experiment.py \
        --nsim 2 --ns 40 --npermutations 10 --functionals linear sine
"""

import argparse
from functools import partial
from itertools import product

import numpy as np

from src.dgp import BernoulliNetwork
from src.helper_functions.simulation_functions import run_simulation
from src.methods.ac_test import MultivariateACTest
from src.methods.cca_test import CanonicalCorrelationTest
from src.methods.distance_correlation_test import DistanceCorrelationTest
from src.metrics import ReturnMetric
from src.solvers.MaMa_uuuuu import pgd_fit_wrapper
from src.solvers.weighted_network import ASE
 

ALPHA = 0.05
K = 3
KY = 1
NOISE_SCALE = 1.0

FUNCTIONALS = {
    "pareto": {"alpha": 0.8},
    "sigmoid": {"global_scale": 10.0},
    "sine": {"global_scale": 10.0, "frequency": 20.0},
    "interaction": {},
    "linear": {},
    "radial": {},
    "tanh_product": {"global_scale": 10.0},
    # "manifold": {},
    # "abs_max": {},
    "max": {},
}

METHODS = (
    "DC/MGC",
    "CCA",
    "AC asymptotic (M=1)",
    "AC asymptotic (M=sqrt(n))",
    "AC adaptive (permutation)",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nsim",
        type=int,
        default=50,
        help="Monte Carlo repetitions per setting (default: 50).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel simulation workers (default: 1).",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=(100, 300),
        help="Network sizes (default: 100 300).",
    )
    parser.add_argument(
        "--npermutations",
        type=int,
        default=100,
        help="Permutations for DC, CCA, and adaptive AC (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Master random seed (default: 1).",
    )
    parser.add_argument(
        "--functionals",
        choices=tuple(FUNCTIONALS),
        nargs="+",
        default=tuple(FUNCTIONALS),
        help="Functional alternatives to include (default: all ten).",
    )
    args = parser.parse_args()

    if args.nsim < 1 or args.jobs < 1:
        parser.error("nsim and jobs must be positive")
    if any(n < 2 for n in args.ns):
        parser.error("network sizes must be at least 2")
    if args.npermutations < 1:
        parser.error("npermutations must be positive")
    return args


def method_designs(args):
    return (
        (
            "DC/MGC",
            partial(
                DistanceCorrelationTest,
                test_method="mgc",
                npermutations=args.npermutations,
                permutation_type="covariate",
            ),
        ),
        (
            "CCA",
            partial(
                CanonicalCorrelationTest,
                npermutations=args.npermutations,
                permutation_type="covariate",
            ),
        ),
        (
            "AC asymptotic (M=1)",
            partial(
                MultivariateACTest,
                M=1,
                test_type="asymptotic",
                use_permutation_coeff=False,
            ),
        ),
        (
            "AC asymptotic (M=sqrt(n))",
            partial(
                MultivariateACTest,
                M=0.5,
                test_type="asymptotic",
                use_permutation_coeff=False,
            ),
        ),
        (
            "AC adaptive (permutation)",
            partial(
                MultivariateACTest,
                adaptive_m=True,
                test_type="permutation",
                npermutations=args.npermutations,
                permutation_type="covariate",
                use_permutation_coeff=False,
            ),
        ),
    )


def build_factorial_design(args):
    design = []
    for n, functional_form in product(args.ns, args.functionals):
        setup = (
            partial(
                BernoulliNetwork,
                k=K,
                ky=KY,
                rdpg=True,
                predictor_distribution="uniform_rdpg",
                center_latent=False,
                functional_form=functional_form,
                function_params=FUNCTIONALS[functional_form],
                rho=None,
                noise_scale=NOISE_SCALE,
            ),
            ASE,
        )
        for method_label, method in method_designs(args):
            design.append(
                {
                    "setup": setup,
                    "method": method,
                    "method_label": method_label,
                    "functional_form": functional_form,
                    "n": n,
                    "k": K,
                    "alpha": ALPHA,
                    "use_true_latent": False,
                    # All methods in this cell reuse one generated network and
                    # one PGD embedding. Method controls are excluded here.
                    "_generation_keys": ("setup", "n"),
                }
            )
    return design


def rejection_rates(results, args):
    pvalues = {
        (n, functional_form, method): []
        for n, functional_form, method in product(
            args.ns, args.functionals, METHODS
        )
    }
    for result in results:
        row = result["args"]
        key = (row["n"], row["functional_form"], row["method_label"])
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
    print(
        f"alpha={ALPHA}, nsim={args.nsim}, k={K}, ky={KY}, "
        f"predictors=gaussian, network=logistic Bernoulli, "
        f"latent=estimated (PGD), noise_scale={NOISE_SCALE:g}, "
        f"permutations={args.npermutations}"
    )
    for n in args.ns:
        print(f"\nn={n}")
        header = f"{'functional':>14}  " + "  ".join(
            f"{method:>30}" for method in METHODS
        )
        print(header)
        for functional_form in args.functionals:
            values = [
                rates[(n, functional_form, method)] for method in METHODS
            ]
            formatted = "  ".join(f"{value:30.3f}" for value in values)
            print(f"{functional_form:>14}  {formatted}")
    print("\nValues are empirical rejection rates (power) at alpha=0.05.")


def main():
    args = parse_args()
    results = run_simulation(
        nsim=args.nsim,
        metrics=[ReturnMetric(only_return="p-value")],
        factorial_design=build_factorial_design(args),
        rng=np.random.default_rng(args.seed),
        parallel=args.jobs > 1,
        n_jobs=args.jobs,
    )
    print_results(rejection_rates(results, args), args)


if __name__ == "__main__":
    main()
