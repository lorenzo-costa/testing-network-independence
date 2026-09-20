"""The example's worker factory stays importable outside a script entry point."""

from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

import run_multiple_networks_example as example
from run_multiple_networks_example import make_network
from src.dgp import GaussianNetwork, BernoulliNetwork


ROOT = Path(__file__).resolve().parents[1]


def test_example_factory_is_defined_in_an_importable_helper_module():
    assert make_network.__module__ == "src.helper_functions.multiple_network_factories"
    assert pickle.loads(pickle.dumps(make_network)) is make_network


@pytest.mark.parametrize("requested,resolved", [(-1, None), (None, None), (2, 2)])
def test_worker_count_uses_runner_convention_without_nested_pools(
    monkeypatch, requested, resolved
):
    captured = {}

    def capture_simulation(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(example, "run_simulation", capture_simulation)
    example.run_experiment(nsim=1, n_jobs=requested)
    assert captured["n_jobs"] == resolved
    assert all(row["n_jobs"] == 1 for row in captured["factorial_design"])


@pytest.mark.parametrize("snr_values", [(0, 0.5, 1, 2), (0, 2)])
def test_example_runs_with_spawn_from_nonimportable_notebook_namespace(snr_values):
    # runpy removes this temporary namespace before the pool starts. A factory
    # defined there cannot be unpickled; an imported factory can.
    code = f"""
import multiprocessing as mp
import runpy

mp.set_start_method("spawn", force=True)
namespace = runpy.run_path("run_multiple_networks_example.py", run_name="notebook_cell")
assert namespace["make_network"].__module__ == "src.helper_functions.multiple_network_factories"
results = namespace["run_experiment"](nsim=1, n=12, npermutations=2, n_jobs=2, snr_values={snr_values!r})
assert len(results) == {2 * len(snr_values)}
assert len(results.groupby(["network", "snr"])) == {2 * len(snr_values)}
assert set(results["snr"]) == set({snr_values!r})
assert results.loc[results["snr"] == 0, "hypothesis"].eq("H0").all()
assert results.loc[results["snr"] > 0, "hypothesis"].eq("H1").all()
assert results.notna().all().all()
assert results["p-value"].between(1 / 3, 1).all()
print("Notebook-style spawn smoke test passed")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Notebook-style spawn smoke test passed" in completed.stdout


@pytest.mark.parametrize("snr_values", [(0,), (0.5, 2.0), (0, 0.5, 1, 2)])
def test_example_has_one_case_per_network_and_snr(monkeypatch, snr_values):
    captured = {}

    def capture_simulation(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(example, "run_simulation", capture_simulation)
    example.run_experiment(nsim=1, snr_values=snr_values)
    assert len(captured["factorial_design"]) == 2 * len(snr_values)
    counts = {target: 0 for target in snr_values}
    for row in captured["factorial_design"]:
        assert row["snr"] in snr_values
        assert row["B"] == (0 if row["snr"] == 0 else None)
        assert row["hypothesis"] == ("H0" if row["snr"] == 0 else "H1")
        counts[row["snr"]] += 1
    assert all(count == 2 for count in counts.values())


def test_default_sweep_has_800_runs_and_reports_each_snr(monkeypatch):
    captured = {}

    def capture_simulation(**kwargs):
        captured.update(kwargs)
        return [
            {
                "args": {**row, "dgp_name": row["setup"][0].args[0].__name__},
                "ReturnMetric": 0.5,
                "ComputeAll": {"Rejection": False},
            }
            for row in kwargs["factorial_design"]
        ]

    monkeypatch.setattr(example, "run_simulation", capture_simulation)
    results = example.run_experiment()
    assert captured["nsim"] == 100
    assert captured["nsim"] * len(captured["factorial_design"]) == 800
    assert len(results.groupby(["network", "snr", "hypothesis"])) == 8
    for _, rows in results.groupby("network"):
        assert rows["snr"].tolist() == [0, 0.5, 1, 2]
        assert rows["hypothesis"].tolist() == ["H0", "H1", "H1", "H1"]


def test_empty_snr_sweep_is_rejected_before_starting_workers(monkeypatch):
    def unexpected_simulation(**kwargs):
        pytest.fail("An empty sweep must not start workers")

    monkeypatch.setattr(example, "run_simulation", unexpected_simulation)
    with pytest.raises(ValueError, match="snr_values"):
        example.run_experiment(snr_values=[])


@pytest.mark.parametrize("network_class", [GaussianNetwork, BernoulliNetwork])
@pytest.mark.parametrize("target", [0, 2.0])
def test_importable_factory_forwards_snr_and_labels_zero_snr_null(
    network_class, target
):
    network = make_network(
        network_class,
        n=10,
        p=2,
        d_x=1,
        d_y=2,
        B=None,
        snr=target,
        rng=np.random.default_rng(306),
    )
    assert network.is_null == (target == 0)
    assert np.sum(network.generate()["B"] ** 2) / 2 == pytest.approx(target)
