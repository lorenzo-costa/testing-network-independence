"""Validation for multiple-network linear-model YAML configurations."""

import numpy as np


def _as_sweep(value, name: str) -> list:
    """Normalize a scalar or list-valued field into a non-empty sweep."""
    if value is None:
        return [None]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} must not be an empty list")
        return value
    return [value]


def _resolve_linear_model_simulation(simulation_cfg: dict) -> dict:
    """Validate and normalize the multiple-network linear-model sweep."""
    if not isinstance(simulation_cfg, dict):
        raise TypeError("simulation must be a mapping")

    required = ("nsim", "seed", "n", "p", "d_x", "d_y")
    missing = [name for name in required if name not in simulation_cfg]
    if missing:
        raise ValueError(
            "linear_model simulation is missing required fields: " + ", ".join(missing)
        )

    signal_parameters = [
        name for name in ("snr", "b_active_network_fraction") if name in simulation_cfg
    ]
    if not signal_parameters:
        raise ValueError(
            "linear_model simulation must specify at least one of snr or "
            "b_active_network_fraction"
        )

    resolved = dict(simulation_cfg)
    for name in (
        "n",
        "p",
        "d_x",
        "d_y",
        *signal_parameters,
        "alpha",
        "edge_var",
    ):
        default = 0.05 if name == "alpha" else 1
        resolved[name] = _as_sweep(
            simulation_cfg.get(name, default), f"simulation.{name}"
        )

    for name in ("nsim", "seed"):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"simulation.{name} must be an integer")
    if resolved["nsim"] < 1:
        raise ValueError("simulation.nsim must be positive")

    for name in ("n", "p", "d_x", "d_y"):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in resolved[name]
        ):
            raise ValueError(f"simulation.{name} must contain positive integer values")
    if "snr" in resolved and any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0
        for value in resolved["snr"]
    ):
        raise ValueError("simulation.snr must contain nonnegative finite values")
    if "b_active_network_fraction" in resolved and any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or not 0 <= value <= 1
        for value in resolved["b_active_network_fraction"]
    ):
        raise ValueError(
            "simulation.b_active_network_fraction must contain finite values in [0, 1]"
        )
    return resolved
