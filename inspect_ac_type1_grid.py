"""Compare asymptotic and permutation AC across n and neighbor count K.

Run: python inspect_ac_type1_grid.py --nsim 100 --jobs 4
"""

import argparse
from functools import partial
from itertools import product

import numpy as np
from src.dgp import BernoulliNetwork
from src.helper_functions.simulation_functions import run_simulation
from src.methods.ac_test import MultivariateACTest
from src.metrics import ReturnMetric
from src.solvers.MaMa_uuuuu import pgd_fit_wrapper


N_VALUES = (100, 300)
K_VALUES = (1, 0.5)
SNR = (0.1, 0.3, 0.5, 0.7)
ALPHA = 0.05
TEST_TYPES = ("asymptotic", "permutation")


sine_params = {
    "weights": [0.5, 0.5, 0.5, 0.5],
    "normalize_index": True,
    "frequency": 5.0,
    "global_scale": 10.0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsim", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--npermutations", type=int, default=200)
    args = parser.parse_args()
    if args.nsim < 1 or args.jobs < 1 or args.npermutations < 1:
        parser.error("nsim, jobs, and npermutations must be positive")

    # Gaussian Z and Y are independent at rho=0. Bounded Z coordinates keep
    # the RDPG inner products in [0, 1].
    setup = (
        partial(
            BernoulliNetwork,
            k=4,
            ky=1,
            rdpg='auto',
            predictor_distribution='gaussian', 
            center_latent=False,
            functional_form = 'sine',
            function_params=sine_params
        ),
        pgd_fit_wrapper,
    )
    factorial_design = []
    for n, snr, K in product(N_VALUES, SNR, K_VALUES):
        for test_type in TEST_TYPES:
            method_kwargs = {
                "M": K,
                "test_type": test_type,
                "permutation_type": "covariate",
                "use_permutation_coeff": False,
            }
            if test_type == "permutation":
                method_kwargs["npermutations"] = args.npermutations
            factorial_design.append({
                "setup": setup,
                "method": partial(MultivariateACTest, **method_kwargs),
                "n": n,
                "snr": snr,
                "alpha": ALPHA,
                "use_true_latent": True,
                "_generation_keys": ("setup", "n", "snr"),
            })

    results = run_simulation(
        nsim=args.nsim,
        metrics=[ReturnMetric(only_return="p-value")],
        factorial_design=factorial_design,
        rng=np.random.default_rng(2),
        parallel=args.jobs > 1,
        n_jobs=args.jobs,
    )

    pvalues = {
        (n, snr, K, test_type): []
        for n, snr, K, test_type in product(
            N_VALUES, SNR, K_VALUES, TEST_TYPES
        )
    }
    for result in results:
        row = result["args"]
        K = row["method"].keywords["M"]
        test_type = row["method"].keywords["test_type"]
        pvalues[(row["n"], row["rho"], K, test_type)].append(
            result["ReturnMetric"]
        )

    print(f"alpha={ALPHA}, nsim={args.nsim}, permutations={args.npermutations}")
    print(" n    snr    K   asymptotic   permutation   difference")
    for n, snr, K in product(N_VALUES, SNR, K_VALUES):
        rates = {}
        for test_type in TEST_TYPES:
            values = pvalues[(n, snr, K, test_type)]
            if len(values) != args.nsim:
                raise RuntimeError(
                    f"Expected {args.nsim} results for {(n, snr, K, test_type)}"
                )
            rates[test_type] = np.count_nonzero(np.asarray(values) < ALPHA) / args.nsim
        print(
            f"{n:3d}  {snr:4.1f}  {K:3d}      {rates['asymptotic']:.3f}"
            f"         {rates['permutation']:.3f}        "
            f"{rates['asymptotic'] - rates['permutation']:+.3f}"
        )
    print("rho=0 gives type I error; rho>0 gives power at alpha=0.05.")


if __name__ == "__main__":
    main()
