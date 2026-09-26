"""Run configured simulation studies, including multiple-network linear models."""

from src.load_config import (
    load_config,
    flatten_args_columns,
    build_factorial_design_multi,
)
from src.helper_functions.simulation_functions import run_simulation

from src.helper_functions.simulation_settings import execution_options

import os
import pandas as pd
from datetime import datetime
import argparse


def main(argv=None):
    """Run the configured factorial design and return the saved CSV path."""
    parser = argparse.ArgumentParser(
        description="Run one or more YAML-configured simulation studies."
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",  # one or more paths
        default=["config.yaml"],
        help="One or more YAML config files (one per experiment type).",
    )
    args = parser.parse_args(argv)

    cfgs = [load_config(p) for p in args.config]
    factorial = build_factorial_design_multi(cfgs)
    print(f"Factorial design has {len(factorial)} rows → {factorial[0]}")

    start = datetime.now()

    out = run_simulation(
        factorial_design=factorial,
        **execution_options(cfgs[0]),
    )

    out = pd.DataFrame(out)
    print(f"Completed simulations in: {datetime.now() - start}")

    # -- Save raw first (guard against column-extraction errors) --------------
    os.makedirs(cfgs[0]["output"]["results_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = cfgs[0]["output"].get("file_prefix", "simulation_results")
    file_name = f"{cfgs[0]['output']['results_dir']}/{prefix}_{timestamp}.csv"
    out.to_csv(file_name, index=False)

    # -- Flatten nested args into columns -------------------------------------
    flatten_args_columns(out)
    out.to_csv(file_name, index=False)
    print(f"Saved {len(out)} rows → {file_name}")
    return file_name


if __name__ == "__main__":
    main()
