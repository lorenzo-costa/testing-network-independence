"""Power experiment for the modulo-one conditional alternative on Bernoulli RDPGs.

Compare AC, MGC, and CCA Model-X tests on shared data and ASE embeddings.
Alternative: Y = (Z + X) mod 1, componentwise.
Null: Y = (Z' + X) mod 1 for an independent draw Z' from the same law.
X is uniform on [0, 1); Z is uniform in the positive orthant of the unit ball.

Examples:
    python power_temp.py
    python power_temp.py --nsim 2 --n 30 --npermutations 20
"""

import argparse
import csv
from functools import partial
from pathlib import Path

import numpy as np

from src.dgp import BernoulliNetwork
from src.helper_functions.simulation_functions import run_simulation
from src.methods.ac_test import MultivariateACTest
from src.methods.cca_test import CanonicalCorrelationTest
from src.methods.distance_correlation_test import DistanceCorrelationTest
from src.metrics import ReturnMetric
from src.solvers.weighted_network import ASE

DEFAULT_NSIM = 100
DEFAULT_SAMPLE_SIZES = (200,)
DEFAULT_NPERMUTATIONS = 99
DEFAULT_JOBS = 6
MIN_SAMPLE_SIZE = 20
SEED = 1
ALPHA = 0.05
LATENT_DIMENSION = 1
RESPONSE_DIMENSION = 1
USE_TRUE_LATENT = False
EMBEDDING_SOLVER = ASE

NETWORK_OPTIONS = {
    "rdpg": True,
    "conditional_modulo": True,
    "x_distribution": "uniform",
    "z_distribution": "uniform_rdpg",
}
MODEL_X_OPTIONS = {"test_type": "model_X", "conditional_generator": "from_dgp"}
AC_METHOD = partial(
    MultivariateACTest, M=0.5, standardize=True,
    use_permutation_coeff=False, **MODEL_X_OPTIONS
)
METHODS = (
    ("AC Model-X (sqrt(n))", AC_METHOD),
    ("MGC Model-X", partial(DistanceCorrelationTest, test_method="mgc", **MODEL_X_OPTIONS)),
    ("CCA Model-X", partial(CanonicalCorrelationTest, **MODEL_X_OPTIONS)),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nsim", type=int, default=DEFAULT_NSIM)
    parser.add_argument("--n", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument("--npermutations", type=int, default=DEFAULT_NPERMUTATIONS,
                        help="Null draws per test (at least 20 for alpha=0.05).")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="Simulation workers; tests each use one worker.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path,
                        default=Path("results") / "power_temp.csv",
                        help="Summary CSV; individual p-values go to <stem>_pvalues.csv.")
    args = parser.parse_args(argv)
    args.n = [args.n] if np.isscalar(args.n) else list(args.n)
    if args.nsim < 1 or args.jobs < 1 or args.npermutations < 20 or args.seed < 0:
        parser.error("--nsim and --jobs must be positive, --seed nonnegative, and --npermutations >= 20.")
    if any(n < MIN_SAMPLE_SIZE for n in args.n):
        parser.error(f"--n must contain sample sizes >= {MIN_SAMPLE_SIZE}.")
    args.n = list(dict.fromkeys(args.n))
    return args


def make_design(args, methods, conditions=("null", "alternative")):
    """Pair methods on each dataset; reuse its embedding via the simulation runner."""
    design = []
    for n in args.n:
        for condition in conditions:
            dgp = partial(
                BernoulliNetwork, **NETWORK_OPTIONS,
                independent_response=(condition == "null"),
            )
            generation = {"setup": (dgp, EMBEDDING_SOLVER), "n": n,
                          "k": LATENT_DIMENSION, "ky": RESPONSE_DIMENSION}
            for label, method in methods:
                design.append({
                    **generation, "_generation_keys": tuple(generation),
                    "condition": condition,
                    "method_label": label, "method": method, "alpha": ALPHA,
                    "npermutations": args.npermutations, "use_true_latent": USE_TRUE_LATENT,
                })
    return design


def experiment_design(args):
    return make_design(args, METHODS)


def result_key(row):
    return (row["n"], row["condition"], row["method_label"])


def wilson_interval(rejections, nsim):
    """95% binomial interval, including when the observed rate is zero or one."""
    z = 1.959963984540054
    rate = rejections / nsim
    denominator = 1 + z**2 / nsim
    center = (rate + z**2 / (2 * nsim)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / nsim + z**2 / (4 * nsim**2)) / denominator
    return max(0., center - half_width), min(1., center + half_width)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args, design):
    results = run_simulation(
        nsim=args.nsim, factorial_design=design,
        metrics=[ReturnMetric(only_return="p-value")], rng=np.random.default_rng(args.seed),
        parallel=args.jobs > 1, n_jobs=args.jobs,
    )
    grouped = {result_key(row): [] for row in design}
    raw = []
    for result in results:
        row = result["args"]
        key = result_key(row)
        pvalue = result["ReturnMetric"]
        if pvalue is None or not np.isfinite(pvalue) or not 0 <= pvalue <= 1:
            raise RuntimeError(f"Invalid p-value for {key}: {pvalue}")
        grouped[key].append(float(pvalue))
        raw.append({"n": row["n"], "condition": row["condition"],
                    "method": row["method_label"], "pvalue": pvalue,
                    "rejected": bool(pvalue < ALPHA), "alpha": ALPHA,
                    "npermutations": args.npermutations, "seed": args.seed})

    summary = []
    print(f"nsim={args.nsim}, alpha={ALPHA}, null draws={args.npermutations}, seed={args.seed}")
    print("Bernoulli RDPG; k=ky=3; estimated latent positions (ASE); uniform X.")
    print("Rates use p < alpha; brackets show 95% Wilson intervals across replicates.")
    print(f"{'n':>5} {'condition':>12}  {'method':25} {'rate':>6}  95% interval")
    for key, values in grouped.items():
        if len(values) != args.nsim:
            raise RuntimeError(f"Expected {args.nsim} p-values for {key}, got {len(values)}")
        n, condition, method = key
        count = int(np.count_nonzero(np.asarray(values) < ALPHA))
        low, high = wilson_interval(count, args.nsim)
        rate = count / args.nsim
        summary.append({"n": n, "condition": condition,
                        "method": method, "nsim": args.nsim, "rejections": count,
                        "rejection_rate": rate, "ci95_low": low, "ci95_high": high,
                        "alpha": ALPHA, "npermutations": args.npermutations, "seed": args.seed})
        print(f"{n:5d} {condition:>12}  {method:25} {rate:6.3f}  [{low:.3f}, {high:.3f}]")
    write_csv(args.output, summary)
    raw_path = args.output.with_name(f"{args.output.stem}_pvalues.csv")
    write_csv(raw_path, raw)
    print(f"Saved {args.output} and {raw_path}")
    return summary


def main():
    args = parse_args()
    print("Power experiment: independent-response null and modulo-one alternative.")
    run_experiment(args, experiment_design(args))


if __name__ == "__main__":
    main()
