#!/usr/bin/env python3
"""Plot Type I error for explicitly zero-covariance copula simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.io import iter_shard_outputs
from src.analysis.processing import PLOT_CHUNKSIZE
from results.visualise_linear_model import (
    NETWORK_LABELS,
    _validate_linear_model_results,
    aggregate_rejection_rates,
    configure_plot_style,
    plot_type_i_error_by_p,
    preprocess_linear_model_results,
    select_asymptotic_null_results,
)


DEFAULT_JOB_ID = "62195697"
DEFAULT_P_ONE_JOB_ID = "62119510"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Type I error for zero-covariance copula shards."
    )
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument(
        "--p-one-job-id",
        default=DEFAULT_P_ONE_JOB_ID,
        help="Earlier zero-covariance job supplying the p=1 panel.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "zero_covariance",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Explicit shard filenames; defaults to files containing the job ID.",
    )
    return parser.parse_args(argv)


def prepare_zero_covariance_results(
    results_dir: Path,
    filenames: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Load zero-covariance shards and normalize them for Type I plots."""
    frames = []
    for chunk in iter_shard_outputs(
        results_dir,
        filenames,
        chunksize=PLOT_CHUNKSIZE,
        usecols=("args", "ComputeAll"),
    ):
        processed = preprocess_linear_model_results(chunk, validate=False)
        if processed.empty:
            continue
        if set(processed["null_target"]) != {"zero_covariance"}:
            raise ValueError(
                "Every row must declare null_target='zero_covariance'."
            )
        if set(processed["hypothesis"]) != {"H0"}:
            raise ValueError("Every zero-covariance row must be labeled H0.")

        # The generic Type I plotting functions use snr=0 as their null-row
        # coordinate. Copula simulations have no SNR, so assign that plotting
        # coordinate without changing the recorded null_target.
        processed["snr"] = 0.0
        frames.append(processed)

    if not frames:
        raise ValueError("The zero-covariance shard files contain no result rows.")

    results = pd.concat(frames, ignore_index=True)
    non_adjacency = results["method"] != "MRQAP"
    latent_modes = set(results.loc[non_adjacency, "use_true_latent"].dropna())
    if len(latent_modes) > 1:
        raise ValueError(
            "Every zero-covariance result set must record at most one latent "
            f"mode for non-MRQAP methods; found {sorted(latent_modes)}."
        )
    # Earlier estimated-latent jobs predate this recorded metadata. Preserve
    # their established interpretation while allowing newer true-latent jobs
    # to carry the mode explicitly.
    has_recorded_latent_mode = bool(latent_modes)
    use_true_latent = latent_modes.pop() if latent_modes else False
    missing_non_adjacency = (
        non_adjacency & results["use_true_latent"].isna()
    )
    if has_recorded_latent_mode and missing_non_adjacency.any():
        raise ValueError(
            f"{int(missing_non_adjacency.sum())} non-MRQAP rows have no "
            "use_true_latent value."
        )
    results["use_true_latent"] = (
        results["use_true_latent"]
        .astype("boolean")
        .fillna(use_true_latent)
        .astype(bool)
    )
    results = select_asymptotic_null_results(results, "split")
    _validate_linear_model_results(results)
    return results


def generate_zero_covariance_figures(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    job_id: str,
) -> list[Path]:
    """Generate one standard Type I error grid for each network model."""
    configure_plot_style()
    results = results[results["method"] != "DC"].copy()
    latent_modes = results["use_true_latent"].dropna().drop_duplicates().tolist()
    if len(latent_modes) != 1:
        raise ValueError(
            "A figure set must contain exactly one latent-position mode; "
            f"found {latent_modes}."
        )
    use_true_latent = bool(latent_modes[0])
    aggregated = aggregate_rejection_rates(results)
    outputs = []
    available = results["dgp_name"].drop_duplicates().tolist()
    networks = [name for name in NETWORK_LABELS if name in available]
    networks.extend(name for name in available if name not in networks)
    for network in networks:
        network_rows = results[results["dgp_name"] == network]
        settings = network_rows[
            ["eps_distribution", "x_network_correlation"]
        ].drop_duplicates()
        if len(settings) != 1:
            raise ValueError(
                f"Expected one plotting setting for {network}; found {len(settings)}."
            )
        setting = settings.iloc[0]
        outputs.append(
            plot_type_i_error_by_p(
                aggregated,
                output_dir,
                network,
                use_true_latent,
                setting["eps_distribution"],
                float(setting["x_network_correlation"]),
                show_setting=False,
                filename_suffix=f"zero_covariance_{job_id}",
            )
        )
    return outputs


def main(argv=None) -> list[Path]:
    args = parse_args(argv)
    if args.files is not None:
        results = prepare_zero_covariance_results(args.results_dir, args.files)
    else:
        result_frames = []
        for job_id in (args.p_one_job_id, args.job_id):
            filenames = sorted(
                path.name
                for path in args.results_dir.glob(f"*{job_id}*shard-*.csv")
            )
            if not filenames:
                raise ValueError(f"No shard files found for job ID {job_id}.")
            result_frames.append(
                prepare_zero_covariance_results(args.results_dir, filenames)
            )
        results = pd.concat(result_frames, ignore_index=True)
    outputs = generate_zero_covariance_figures(
        results,
        args.output_dir,
        job_id=args.job_id,
    )
    for output in outputs:
        print(output)
    return outputs


if __name__ == "__main__":
    main()
