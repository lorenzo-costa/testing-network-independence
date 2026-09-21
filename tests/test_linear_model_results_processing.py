from pathlib import Path

import pandas as pd
import pytest

from results.results_processing import (
    combine_shard_outputs,
    iter_shard_outputs,
    process_shard_results,
)
from results.visualise_linear_model import (
    aggregate_frobenius_errors,
    aggregate_rejection_rates,
    prepare_linear_model_results,
)


def _row(
    n,
    p,
    snr,
    rejection,
    method="RVTest",
    approximation="permutation",
    gamma=None,
    use_true_latent=False,
    frobenius_y=0.2,
):
    args = {
        "n": n,
        "p": p,
        "d_x": 5,
        "d_y": 4,
        "snr": snr,
        "x_network_correlation": 0.5,
        "eps_distribution": "student_t_3",
        "hypothesis": "H0" if snr == 0 else "H1",
        "alpha": 0.05,
        "npermutations": 400,
        "use_true_latent": use_true_latent,
        "B": 0 if snr == 0 else None,
        "method": method,
        "permutation_type": "latent",
        "test_method": "mgc" if method == "DistanceCorrelationTest" else "NA",
        "dgp_name": "GaussianNetwork_multiple_networks",
        "solver": "ASE",
    }
    if method == "RVTest":
        args["approximation"] = approximation
    if gamma is not None:
        args["gamma"] = gamma
    return {
        "args": args,
        "ComputeAll": {
            "Rejection": rejection,
            "FalseRejection": snr == 0 and rejection,
            "TrueRejection": snr > 0 and rejection,
            "RelativeFrobeniusNorm_Y": frobenius_y,
            "RelativeFrobeniusNorm_X_1": 0.3,
            "RelativeFrobeniusNorm_X_global": 0.25,
            "ProcrustesDistance_Y": 0.1,
            "ProcrustesDistance_X_1": 0.15,
            "ProcrustesDistance_X_global": 0.12,
        },
    }


def _write_shards(tmp_path: Path):
    names = [
        "linear_model_results_123_shard-000-of-003.csv",
        "linear_model_results_123_shard-001-of-003.csv",
        "linear_model_results_123_shard-002-of-003.csv",
    ]
    rows = [
        _row(50, 5, 0, False),
        _row(100, 5, 0.5, True, method="CanonicalCorrelationTest", gamma=0.7),
        _row(200, 10, 1, True, method="DistanceCorrelationTest"),
    ]
    for name, row in zip(names, rows):
        pd.DataFrame([row]).to_csv(tmp_path / name, index=False)
    return names


def test_combine_and_process_linear_model_shards(tmp_path):
    names = _write_shards(tmp_path)

    combined = combine_shard_outputs(tmp_path, names)
    assert len(combined) == 3
    assert combined["shard_index"].tolist() == [0, 1, 2]
    assert combined["num_shards"].tolist() == [3, 3, 3]

    processed = process_shard_results(tmp_path, names)
    assert processed["n"].tolist() == [50, 100, 200]
    assert processed["p"].tolist() == [5, 5, 10]
    assert processed["d_x"].tolist() == [5, 5, 5]
    assert processed["d_y"].tolist() == [4, 4, 4]
    assert processed["snr"].tolist() == [0.0, 0.5, 1.0]
    assert processed["x_network_correlation"].tolist() == [0.5, 0.5, 0.5]
    assert processed["eps_distribution"].tolist() == ["student_t_3"] * 3
    assert processed["npermutations"].tolist() == [400, 400, 400]
    assert processed["permutation_type"].tolist() == ["latent"] * 3
    assert processed["method"].tolist() == [
        "RVTest_permutation",
        "CCA",
        "DC",
    ]
    assert processed["cca_gamma"].tolist() == ["NA", 0.7, "NA"]
    for column in (
        "RelativeFrobeniusNorm_Y",
        "RelativeFrobeniusNorm_X_1",
        "RelativeFrobeniusNorm_X_global",
        "ProcrustesDistance_Y",
        "ProcrustesDistance_X_1",
        "ProcrustesDistance_X_global",
    ):
        assert column in processed

    aggregate = aggregate_rejection_rates(processed)
    assert aggregate["replicates"].tolist() == [1, 1, 1]


def test_chunked_shard_reader_retains_shard_metadata(tmp_path):
    names = _write_shards(tmp_path)

    chunks = list(
        iter_shard_outputs(
            tmp_path,
            names,
            chunksize=1,
            usecols=("args", "ComputeAll"),
        )
    )

    assert len(chunks) == 3
    assert [chunk["shard_index"].iloc[0] for chunk in chunks] == [0, 1, 2]
    assert all(chunk["num_shards"].iloc[0] == 3 for chunk in chunks)


def test_plot_aggregations_separate_latent_modes_and_y_frobenius_error():
    base = {
        "dgp_name": "GaussianNetwork",
        "p": 5,
        "snr": 0.5,
        "n": 100,
        "method": "CCA",
        "alpha": 0.05,
    }
    results = pd.DataFrame(
        [
            {
                **base,
                "use_true_latent": False,
                "Rejection": 1,
                "RelativeFrobeniusNorm_Y": 0.4,
            },
            {
                **base,
                "use_true_latent": False,
                "Rejection": 0,
                "RelativeFrobeniusNorm_Y": 0.2,
            },
            {
                **base,
                "use_true_latent": True,
                "Rejection": 1,
                "RelativeFrobeniusNorm_Y": 0.0,
            },
        ]
    )

    rejection = aggregate_rejection_rates(results).sort_values("use_true_latent")
    frobenius = aggregate_frobenius_errors(results).sort_values("use_true_latent")

    assert rejection["use_true_latent"].tolist() == [False, True]
    assert rejection["rejection_rate"].tolist() == [0.5, 1.0]
    assert frobenius["frobenius_error"].tolist() == pytest.approx([0.3, 0.0])
    assert frobenius["replicates"].tolist() == [2, 1]


def test_asymptotic_shards_keep_only_rv_rows(tmp_path):
    names = [
        "linear_model_asymptotic_results_123_shard-000-of-003.csv",
        "linear_model_asymptotic_results_123_shard-001-of-003.csv",
        "linear_model_asymptotic_results_123_shard-002-of-003.csv",
    ]
    rows = [
        _row(50, 5, 0.5, True, approximation="asymptotic"),
        _row(50, 5, 0.5, True, method="CanonicalCorrelationTest"),
        _row(50, 5, 0.5, True, method="DistanceCorrelationTest"),
    ]
    for name, row in zip(names, rows):
        pd.DataFrame([row]).to_csv(tmp_path / name, index=False)

    results = prepare_linear_model_results(
        tmp_path,
        names,
        include_methods=("RVTest_asymptotic",),
    )

    assert len(results) == 1
    assert results["method"].tolist() == ["RVTest_asymptotic"]
    assert results["Rejection"].tolist() == [1]


def test_combine_shards_rejects_an_incomplete_set(tmp_path):
    names = _write_shards(tmp_path)
    with pytest.raises(ValueError, match="Incomplete shard set"):
        combine_shard_outputs(tmp_path, names[:2])
