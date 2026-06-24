"""
run_simulation_script.py  —  Main copula study
Uses: config.yaml
"""

from src.load_config import (
    load_config,
    build_factorial_design,
    flatten_args_columns,
    build_factorial_design_multi,
)
from src.helper_functions.simulation_functions import run_simulation

import os
import pandas as pd
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the main simulation script for the copula study."
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",  # one or more paths
        default=["config.yaml"],
        help="One or more YAML config files (one per experiment type).",
    )
    args = parser.parse_args()

    cfgs = [load_config(p) for p in args.config]
    factorial = build_factorial_design_multi(cfgs)
    sim = cfgs[0]["simulation"]

    start = datetime.now()

    # -- H1 run ---------------------------------------------------------------
    out = run_simulation(
        nsim=sim["nsim"],
        metrics=cfgs[0]["metrics"],
        factorial_design=factorial,
        rng=cfgs[0]["rng"],
        parallel=True,
    )

    out = pd.DataFrame(out)
    print(f"Completed H1 simulations in: {datetime.now() - start}")

    # # -- H0 run ---------------------------------------------------------------
    # rng_null = __import__("numpy").random.default_rng(sim["seed"])
    # out0 = run_simulation(
    #     nsim=sim["nsim"],
    #     metrics=cfg["metrics"],
    #     factorial_design=factorial_h0,
    #     rng=rng_null,
    #     parallel=True,
    # )

    # out0 = pd.DataFrame(out0)

    # print(f"Total simulation time: {datetime.now() - start}")

    # results = pd.concat([out, out0], ignore_index=True)

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
