"""Power experiment for a continuous conditional copula on Bernoulli RDPGs.

Compare AC, MGC, and CCA Model-X tests on shared data and ASE embeddings.
The conditional Z-Y correlation is zero under the null and 0.5 under the
alternative. Correlations of Z and Y with X each range over {0, 0.5}.
X and Y have standard Gaussian marginals; Z is uniform on [0, 1].
Model-X null draws use the copula sampler's exact Y | X distribution.

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

DEFAULT_NSIM = 50
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
    "continuous_conditional_copula": "gaussian",
    "marginals": {"x": "gaussian", "z": "uniform 0 1", "y": "gaussian"},
}
CORRELATIONS_WITH_X = (0.0, 0.5)
ALTERNATIVE_CONDITIONAL_CORRELATION = 0.5
MODEL_X_OPTIONS = {"test_type": "model_X", "conditional_generator": "from_dgp"}


AC_METHOD = partial(
    MultivariateACTest, M=0.5, standardize=False,
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
        for correlation_zx in CORRELATIONS_WITH_X:
            for correlation_yx in CORRELATIONS_WITH_X:
                for condition in conditions:
                    dgp = partial(
                        BernoulliNetwork, **NETWORK_OPTIONS,
                        correlation_zx=correlation_zx,
                        correlation_yx=correlation_yx,
                        conditional_correlation_zy=(
                            0.0 if condition == "null"
                            else ALTERNATIVE_CONDITIONAL_CORRELATION
                        ),
                    )
                    generation = {"setup": (dgp, EMBEDDING_SOLVER), "n": n,
                                  "k": LATENT_DIMENSION, "ky": RESPONSE_DIMENSION}
                    for label, method in methods:
                        design.append({
                            **generation, "_generation_keys": tuple(generation),
                            "condition": condition,
                            "correlation_zx": correlation_zx,
                            "correlation_yx": correlation_yx,
                            "method_label": label, "method": method, "alpha": ALPHA,
                            "npermutations": args.npermutations,
                            "use_true_latent": USE_TRUE_LATENT,
                        })
    return design


def experiment_design(args):
    return make_design(args, METHODS)


def result_key(row):
    return (row["n"], row["correlation_zx"], row["correlation_yx"],
            row["condition"], row["method_label"])


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
        raw.append({"n": row["n"], "correlation_zx": row["correlation_zx"],
                    "correlation_yx": row["correlation_yx"], "condition": row["condition"],
                    "method": row["method_label"], "pvalue": pvalue,
                    "rejected": bool(pvalue < ALPHA), "alpha": ALPHA,
                    "npermutations": args.npermutations, "seed": args.seed})

    summary = []
    print(f"nsim={args.nsim}, alpha={ALPHA}, null draws={args.npermutations}, seed={args.seed}")
    print(
        f"Bernoulli RDPG; k={LATENT_DIMENSION}, ky={RESPONSE_DIMENSION}; "
        "estimated latent positions (ASE); uniform Z; exact conditional Y draws."
    )
    print("Rates use p < alpha; brackets show 95% Wilson intervals across replicates.")
    print(f"{'n':>5} {'rho_zx':>6} {'rho_yx':>6} {'condition':>12}  {'method':25} {'rate':>6}  95% interval")
    for key, values in grouped.items():
        if len(values) != args.nsim:
            raise RuntimeError(f"Expected {args.nsim} p-values for {key}, got {len(values)}")
        n, correlation_zx, correlation_yx, condition, method = key
        count = int(np.count_nonzero(np.asarray(values) < ALPHA))
        low, high = wilson_interval(count, args.nsim)
        rate = count / args.nsim
        summary.append({"n": n, "correlation_zx": correlation_zx,
                        "correlation_yx": correlation_yx, "condition": condition,
                        "method": method, "nsim": args.nsim, "rejections": count,
                        "rejection_rate": rate, "ci95_low": low, "ci95_high": high,
                        "alpha": ALPHA, "npermutations": args.npermutations, "seed": args.seed})
        print(f"{n:5d} {correlation_zx:6.1f} {correlation_yx:6.1f} "
              f"{condition:>12}  {method:25} {rate:6.3f}  [{low:.3f}, {high:.3f}]")
    write_csv(args.output, summary)
    raw_path = args.output.with_name(f"{args.output.stem}_pvalues.csv")
    write_csv(raw_path, raw)
    print(f"Saved {args.output} and {raw_path}")
    return summary


def main():
    args = parse_args()
    print("Power experiment: continuous conditional copula null and alternative.")
    run_experiment(args, experiment_design(args))


if __name__ == "__main__":
    main()
