"""Small continuous-X Gaussian-copula power experiment on Bernoulli RDPGs.

Compare AC, MGC, and CCA Model-X tests on shared data and ASE embeddings.
Residual Z-Y correlations apply to matching coordinates, conditional on X.
Zero residual correlation estimates conditional Type I error.
X and residual correlations are crossed factorially (all combinations).

Examples:
    python power_temp.py
    python power_temp.py --rho 0.5 --x-correlations 0 0.3 0.6
    python power_temp.py --nsim 2 --n 30 --x-correlation 0 0.5 --rho 0 0.5 --npermutations 20
"""

import argparse
import csv
from functools import partial
from itertools import product
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
DEFAULT_SAMPLE_SIZES = (200)
DEFAULT_RHOS = 0.3
DEFAULT_NPERMUTATIONS = 99
DEFAULT_JOBS = 6
MIN_SAMPLE_SIZE = 20
SEED = 1
ALPHA = 0.05
LATENT_DIMENSION = 3
RESPONSE_DIMENSION = 3
USE_TRUE_LATENT = False
EMBEDDING_SOLVER = ASE
DEFAULT_X_CORRELATIONS = 0

# Z lies in [0, 1/sqrt(k)] so every RDPG dot product is a probability.
MARGINALS = {
    "x": f"cauchy",
    "z": f"uniform 0 {1 / np.sqrt(LATENT_DIMENSION)}",
    "y": "cauchy",
}

NETWORK_OPTIONS = {
    "rdpg": True,
    "continuous_conditional_copula": "gaussian",
    "center_latent": False,
    "marginals": MARGINALS,
}
MODEL_X_OPTIONS = {"test_type": "model_X", "conditional_generator": "from_dgp"}
AC_METHOD = partial(
    MultivariateACTest, M=0.5, standardize=True,
    use_permutation_coeff=True, **MODEL_X_OPTIONS
)
METHODS = (
    ("AC Model-X (sqrt(n))", AC_METHOD),
    ("MGC Model-X", partial(DistanceCorrelationTest, test_method="mgc", **MODEL_X_OPTIONS)),
    ("CCA Model-X", partial(CanonicalCorrelationTest, **MODEL_X_OPTIONS)),
)


def parse_args(argv=None, *, validity=False):
    parser = argparse.ArgumentParser(
        description=("Conditional-null validity: AC Model-X versus global DC permutation."
                     if validity else __doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nsim", type=int, default=DEFAULT_NSIM)
    parser.add_argument("--n", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES)
    if validity:
        parser.add_argument("--x-correlation", "--x-correlations", type=float, nargs="+", default=(0.0, 0.6),
                            help="Normal-coordinate associations of both Z and Y with X.")
    else:
        parser.add_argument("--rho", type=float, nargs="+", default=DEFAULT_RHOS,
                            help="Residual Z-Y correlations on matching coordinates given X.")
        parser.add_argument("--x-correlation", "--x-correlations", type=float, nargs="+", default=DEFAULT_X_CORRELATIONS,
                            help="X associations, crossed with every --rho value.")
    parser.add_argument("--npermutations", type=int, default=DEFAULT_NPERMUTATIONS,
                        help="Null draws per test (at least 20 for alpha=0.05).")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="Simulation workers; tests each use one worker.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path,
                        default=Path("results") / ("validity_temp.csv" if validity else "power_temp.csv"),
                        help="Summary CSV; individual p-values go to <stem>_pvalues.csv.")
    args = parser.parse_args(argv)
    # Editable defaults may be scalars (e.g. DEFAULT_RHOS = 0.5), whereas
    # command-line sweeps arrive as lists. Normalize both before validation.
    for name in ("n", "x_correlation", *(() if validity else ("rho",))):
        value = getattr(args, name)
        setattr(args, name, [value] if np.isscalar(value) else list(value))
    if args.nsim < 1 or args.jobs < 1 or args.npermutations < 20 or args.seed < 0:
        parser.error("--nsim and --jobs must be positive, --seed nonnegative, and --npermutations >= 20.")
    if any(n < MIN_SAMPLE_SIZE for n in args.n):
        parser.error(f"--n must contain sample sizes >= {MIN_SAMPLE_SIZE}.")
    args.n = list(dict.fromkeys(args.n))
    correlations = args.x_correlation if validity else [*args.rho, *args.x_correlation]
    if any(not np.isfinite(c) or abs(c) >= 1 for c in correlations):
        parser.error("Correlations must be finite and strictly between -1 and 1.")
    args.x_correlation = list(dict.fromkeys(args.x_correlation))
    if not validity:
        args.rho = list(dict.fromkeys(args.rho))
    return args


def make_design(args, methods, associations):
    """Pair methods on each dataset; reuse its embedding via the simulation runner."""
    design = []
    for n, (x_correlation, residual_correlation) in product(args.n, associations):
        dgp = partial(
            BernoulliNetwork, **NETWORK_OPTIONS,
            correlation_zx=x_correlation, correlation_yx=x_correlation,
            conditional_correlation_zy=(residual_correlation
                                       * np.eye(LATENT_DIMENSION, RESPONSE_DIMENSION)),
        )
        generation = {"setup": (dgp, EMBEDDING_SOLVER), "n": n,
                      "k": LATENT_DIMENSION, "ky": RESPONSE_DIMENSION}
        for label, method in methods:
            design.append({
                **generation, "_generation_keys": tuple(generation),
                "x_correlation": x_correlation,
                "residual_correlation": residual_correlation,
                "method_label": label, "method": method, "alpha": ALPHA,
                "npermutations": args.npermutations, "use_true_latent": USE_TRUE_LATENT,
            })
    return design


def experiment_design(args):
    return make_design(args, METHODS, list(product(args.x_correlation, args.rho)))


def result_key(row):
    return (row["n"], row["x_correlation"], row["residual_correlation"], row["method_label"])


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
        raw.append({"n": row["n"], "x_correlation": row["x_correlation"],
                    "residual_correlation": row["residual_correlation"],
                    "method": row["method_label"], "pvalue": pvalue,
                    "rejected": bool(pvalue < ALPHA), "alpha": ALPHA,
                    "npermutations": args.npermutations, "seed": args.seed})

    summary = []
    print(f"nsim={args.nsim}, alpha={ALPHA}, null draws={args.npermutations}, seed={args.seed}")
    print("Bernoulli RDPG; k=ky=3; estimated latent positions (ASE); scalar Gaussian X.")
    print("Rates use p < alpha; brackets show 95% Wilson intervals across replicates.")
    print(f"{'n':>5} {'X corr':>7} {'resid corr':>10}  {'method':25} {'rate':>6}  95% interval")
    for key, values in grouped.items():
        if len(values) != args.nsim:
            raise RuntimeError(f"Expected {args.nsim} p-values for {key}, got {len(values)}")
        n, x_corr, rho, method = key
        count = int(np.count_nonzero(np.asarray(values) < ALPHA))
        low, high = wilson_interval(count, args.nsim)
        rate = count / args.nsim
        summary.append({"n": n, "x_correlation": x_corr, "residual_correlation": rho,
                        "method": method, "nsim": args.nsim, "rejections": count,
                        "rejection_rate": rate, "ci95_low": low, "ci95_high": high,
                        "alpha": ALPHA, "npermutations": args.npermutations, "seed": args.seed})
        print(f"{n:5d} {x_corr:7g} {rho:10g}  {method:25} {rate:6.3f}  [{low:.3f}, {high:.3f}]")
    write_csv(args.output, summary)
    raw_path = args.output.with_name(f"{args.output.stem}_pvalues.csv")
    write_csv(raw_path, raw)
    print(f"Saved {args.output} and {raw_path}")
    return summary


def main():
    args = parse_args()
    print("Power experiment: residual correlation zero is the conditional-null control.")
    run_experiment(args, experiment_design(args))


if __name__ == "__main__":
    main()
