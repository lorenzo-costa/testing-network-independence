"""Current interface contracts that structural refactors must preserve."""

from copy import deepcopy
from functools import partial
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

from src.load_config import build_factorial_design, load_config
from src.metrics import (
    ComputeAll,
    RelativeFrobeniusNorm,
    RobustRelativeProcrustesDistance,
)
from src.methods import RVTest
from src.solvers.MaMa_uuuuu import pgd_fit_wrapper
from src.analysis.processing import preprocess_results


def test_joint_grid_order_and_null_metadata_are_independent_of_research_yaml(tmp_path):
    path = tmp_path / "small.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_type": "linear_model",
                "simulation": {
                    "nsim": 1,
                    "seed": 12,
                    "n": [10],
                    "p": [1, 3],
                    "d_x": [1],
                    "d_y": [2],
                    "snr": [0.25, 0.5],
                    "b_active_network_fraction": [0, 0.5],
                },
                "methods": {
                    "list": [{"name": "RVTest", "kwargs": {"n_jobs": 1}}],
                    "npermutations": [3],
                    "use_true_latent": [True],
                },
                "setups": [{"dgp": "GaussianNetwork", "solver": "ASE"}],
                "metrics": {"compute_all": True},
            }
        )
    )
    rows = build_factorial_design(load_config(path))
    assert [
        (
            row["p"],
            row["requested_snr"],
            row["b_active_network_fraction"],
            row["snr"],
            row["B"],
            row["hypothesis"],
        )
        for row in rows
    ] == [
        (1, 0.25, 0, 0, 0, "H0"),
        (1, 0.25, 0.5, 0.25, None, "H1"),
        (1, 0.5, 0, 0, 0, "H0"),
        (1, 0.5, 0.5, 0.5, None, "H1"),
        (3, 0.25, 0, 0, 0, "H0"),
        (3, 0.25, 0.5, 0.25, None, "H1"),
        (3, 0.5, 0, 0, 0, "H0"),
        (3, 0.5, 0.5, 0.5, None, "H1"),
    ]
    assert all(row["method"].func is RVTest for row in rows)
    assert rows[0]["method"].keywords == {"n_jobs": 1}


@pytest.mark.parametrize("missing_truth", [False, True])
@pytest.mark.parametrize("gram_matrix", [False, True])
def test_combined_recovery_metrics_equal_independent_calls(missing_truth, gram_matrix):
    rng = np.random.default_rng(123)
    x = [rng.normal(size=(12, 2)), rng.normal(size=(12, 2))]
    y = rng.normal(size=(12, 3))
    result = {
        "estimated_latent": {
            "Y": y + 0.2,
            "X_blocks": [v + 0.1 for v in x],
            "X": np.zeros((12, 4)),
        },
        "true_latent": None if missing_truth else {"Y": y, "X_blocks": x},
        "reject_null": False,
    }
    combined = ComputeAll(gram_matrix)(result, is_null=True)
    for prefix, metric in [
        ("RelativeFrobeniusNorm", RelativeFrobeniusNorm(gram_matrix)),
        ("ProcrustesDistance", RobustRelativeProcrustesDistance()),
    ]:
        for name, value in metric(result).items():
            np.testing.assert_equal(combined[f"{prefix}_{name}"], value)


def test_processing_owns_its_frame_and_preserves_nested_input():
    config = {
        "n": 12,
        "p": 2,
        "d_x": 1,
        "d_y": 1,
        "snr": 0.5,
        "method": "RVTest",
        "dgp_name": "GaussianNetwork",
    }
    raw = pd.DataFrame(
        {"args": [config], "ComputeAll": [{"Rejection": True}], "n": [-1]}, index=[7]
    )
    before = deepcopy(raw.to_dict())
    processed = preprocess_results(raw)
    assert raw.to_dict() == before
    assert processed.iloc[0]["n"] == 12
    processed.loc[processed.index[0], "n"] = 999
    assert raw.to_dict() == before


def test_serialized_public_callable_locations_survive_internal_extraction():
    assert RVTest.__module__ == "src.methods.rv_test"
    assert pgd_fit_wrapper.__module__ == "src.solvers.MaMa_uuuuu"
    for factory in [RVTest, pgd_fit_wrapper, partial(pgd_fit_wrapper, backend="numpy")]:
        restored = pickle.loads(pickle.dumps(factory))
        if isinstance(factory, partial):
            assert restored.func is factory.func
            assert restored.keywords == factory.keywords
        else:
            assert restored is factory
