"""Type I error and power of asymptotic AC across n and neighbor count K.

Run: python inspect_ac_type1_grid.py --nsim 50 --jobs 4
"""

import argparse
from functools import partial
from itertools import product

import numpy as np
from scipy.stats import binomtest

from src.dgp import BernoulliNetwork
from src.helper_functions.simulation_functions import run_simulation
from src.methods.ac_test import MultivariateACTest
from src.metrics import ReturnMetric
from src.solvers.weighted_network import ASE


N_VALUES = (100, 200, 300)
K_VALUES = (1, 10, 50)
RHO_VALUES = (0.0, 0.1, 0.3, 0.5)
ALPHA = 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsim", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.nsim < 1 or args.jobs < 1:
        parser.error("nsim and jobs must be positive")

    # Gaussian Z and Y are independent at rho=0. The logistic link turns
    # Gaussian latent inner products into valid Bernoulli edge probabilities.
    setup = (
        partial(
            BernoulliNetwork,
            k=3,
            ky=2,
            rdpg=True,
            copula_model="gaussian",
            marginals={"z": "uniform 0 0.5773502692", "y": "gaussian"},
            center_latent=False,
        ),
        ASE,
    )
    factorial_design = [
        {
            "setup": setup,
            "method": partial(
                MultivariateACTest,
                M=K,
                test_type="asymptotic",
                use_permutation_coeff=True,
            ),
            "n": n,
            "rho": rho,
            "alpha": ALPHA,
            "use_true_latent": True,
            "_generation_keys": ("setup", "n", "rho"),
        }
        for n, rho, K in product(N_VALUES, RHO_VALUES, K_VALUES)
    ]

    results = run_simulation(
        nsim=args.nsim,
        metrics=[ReturnMetric(only_return="p-value")],
        factorial_design=factorial_design,
        rng=np.random.default_rng(2),
        parallel=args.jobs > 1,
        n_jobs=args.jobs,
    )

    pvalues = {
        (n, rho, K): []
        for n, rho, K in product(N_VALUES, RHO_VALUES, K_VALUES)
    }
    for result in results:
        row = result["args"]
        K = row["method"].keywords["M"]
        pvalues[(row["n"], row["rho"], K)].append(result["ReturnMetric"])

    print(" n    rho    K   reject/n   rate    95% binomial CI")
    for (n, rho, K), values in pvalues.items():
        if len(values) != args.nsim:
            raise RuntimeError(f"Expected {args.nsim} results for {(n, rho, K)}")
        rejected = np.count_nonzero(np.asarray(values) < ALPHA)
        ci = binomtest(rejected, len(values)).proportion_ci()
        print(
            f"{n:3d}  {rho:4.1f}  {K:3d}   {rejected:2d}/{len(values):2d}"
            f"     {rejected / len(values):.3f}   [{ci.low:.3f}, {ci.high:.3f}]"
        )
    print("rho=0 gives type I error; rho>0 gives power at alpha=0.05.")


if __name__ == "__main__":
    main()
