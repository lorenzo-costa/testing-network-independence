"""Factorial rows for the multiple-network linear model."""

import numpy as np

from .sweeps import expand_sweeps


def build_linear_model_rows(cfg):
    sim = cfg["simulation"]
    mth = cfg["methods"]
    sets = cfg["setups"]
    signal_parameters = [
        name for name in ("snr", "b_active_network_fraction") if name in sim
    ]
    sweeps = [
        ("setup", sets),
        ("method", mth["list"]),
        ("n", sim["n"]),
        ("p", sim["p"]),
        ("d_x", sim["d_x"]),
        ("d_y", sim["d_y"]),
        *[(name, sim[name]) for name in signal_parameters],
        ("alpha", sim["alpha"]),
        ("edge_var", sim["edge_var"]),
        ("npermutations", mth["npermutations"]),
    ]
    if mth["use_true_latent"] is not None:
        sweeps.append(("use_true_latent", mth["use_true_latent"]))

    rows = expand_sweeps(sweeps)
    for row in rows:
        fraction = row.get("b_active_network_fraction", 1)
        active_count = int(np.floor(fraction * row["p"] + 0.5))
        is_null = active_count == 0 or row.get("snr") == 0
        if len(signal_parameters) == 2:
            # Preserve the requested sweep coordinate when no active block
            # remains: its attainable SNR is zero, regardless of the target.
            row["requested_snr"] = row["snr"]
            if is_null:
                row["snr"] = 0
        row["B"] = 0 if is_null else None
        row["hypothesis"] = "H0" if is_null else "H1"
    return rows
