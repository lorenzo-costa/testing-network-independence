from src.load_config import load_config
 
from src.metrics import ComputeAll
from src.helper_functions.simulation_functions import run_simulation
from src.helper_functions.analyse_functions import aggregate_results
 
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
import os
 
 
if __name__ == "__main__":
 
    # ── Load all parameters from config.yaml ─────────────────────────────────
    cfg = load_config("config.yaml")
 
    sim      = cfg["simulation"]
    methods  = cfg["methods"]
    setups   = cfg["setups"]
    null     = cfg["null_setups"]
    metrics  = cfg["metrics"]
    rng      = cfg["rng"]
    out_dir  = cfg["output"]["results_dir"]
    
    # save setup
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f"{out_dir}/simulation_results_{timestamp}.csv"
 
    # ── Build factorial design (H1)  ─────────────────────────────────
    param_names = [
        "setup", "method", "n", "k", "alpha", "marginals",
        "rho", "edge_var", "approximation", "npermutations",
        "df", "use_true_latent",
    ]
 
    param_values = product(
        setups,
        methods["list"],
        sim["n"],
        sim["k"],
        sim["alpha"],
        sim["marginals"],
        sim["rho"],
        sim["edge_var"],
        methods["approximation"],
        methods["npermutations"],
        methods["df"],
        methods["use_true_latent"],
    )
 
    factorial_design = [dict(zip(param_names, v)) for v in param_values]
 
    out = run_simulation(
        nsim=sim["nsim"],
        metrics=metrics,
        factorial_design=factorial_design,
        rng=rng,
        parallel=True,
    )
    
    out = pd.DataFrame(out)
    
    # save this first in case next one crashes
    out.to_csv(file_name, index=False)
    
 
    # ── Build factorial design (H0) ─────────────────────────────────
    rng2 = np.random.default_rng(sim["seed"])   # fresh seeded RNG for null runs
 
    param_values2 = product(
        null["setups"],
        methods["list"],
        sim["n"],
        sim["k"],
        sim["alpha"],
        sim["marginals"],
        null["rho"],                             # rho = 0.0
        sim["edge_var"],
        methods["approximation"],
        methods["npermutations"],
        methods["df"],
        methods["use_true_latent"],
    )
 
    factorial_design2 = [dict(zip(param_names, v)) for v in param_values2]
 
    out2 = run_simulation(
        nsim=sim["nsim"],
        metrics=metrics,
        factorial_design=factorial_design2,
        rng=rng2,
        parallel=True,
    )
 
    # ── Combine & save ────────────────────────────────────────────────────────
    results = pd.concat([pd.DataFrame(out), pd.DataFrame(out2)], ignore_index=True)
 
    # Save raw first (guard against column-extraction errors below)
    results.to_csv(file_name, index=False)
 
    # ── Flatten nested 'args' dict into columns ───────────────────────────────
    results["n"]            = results["args"].apply(lambda x: x["n"])
    results["k"]            = results["args"].apply(lambda x: x["k"])
    results["edge_var"]     = results["args"].apply(lambda x: x.get("edge_var", "NA"))
    results["approximation"]= results["args"].apply(lambda x: x.get("approximation", "NA"))
    results["dgp"]          = results["args"].apply(lambda x: x.get("dgp_name", "NA"))
    results["solver"]       = results["args"].apply(lambda x: x.get("solver", "NA"))
    results["rho"]          = results["args"].apply(lambda x: x.get("rho", "NA"))
    results["method"]       = results["args"].apply(lambda x: x.get("method_name", "NA"))
    results["marginals"]    = results["args"].apply(
        lambda x: x.get("marginals").name if hasattr(x.get("marginals"), "name") else "NA"
    )
 
    results.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")
