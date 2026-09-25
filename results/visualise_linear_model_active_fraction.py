#!/usr/bin/env python3
"""Plot the linear-model experiment indexed by active network fraction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from results.visualise_linear_model import (  # noqa: E402
    generate_linear_model_figures,
    prepare_active_fraction_results,
)


RESULT_FILES = (
    "linear_model_active_fraction_results_61834176_shard-000-of-005.csv",
    "linear_model_active_fraction_results_61834176_shard-001-of-005.csv",
    "linear_model_active_fraction_results_61834176_shard-002-of-005.csv",
    "linear_model_active_fraction_results_61834176_shard-003-of-005.csv",
    "linear_model_active_fraction_results_61834176_shard-004-of-005.csv",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot power and type-I error by active network fraction."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "linear_model_active_fraction_figures",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=RESULT_FILES,
    )
    parser.add_argument(
        "--exclude-n",
        nargs="*",
        type=int,
        default=(500,),
        help="Network sizes to exclude from the figures (default: 500).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> list[Path]:
    args = parse_args(argv)
    results = prepare_active_fraction_results(args.results_dir, args.files)
    results = results[~results["n"].isin(args.exclude_n)].copy()
    outputs = generate_linear_model_figures(
        results,
        args.output_dir,
        asymptotic_null="independence",
        effect_label=r"$f_{\mathrm{active}}$",
        testing_only=True,
    )
    print(f"Saved {len(outputs)} figures to {args.output_dir}")
    return outputs


if __name__ == "__main__":
    main()
