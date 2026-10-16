"""Regression tests for one dataset and embedding per replicate/data group."""

from functools import partial
from pathlib import Path
import time

import numpy as np
import pytest

from src.helper_functions.simulation_functions import _groups, run_simulation, shard_row_count
from src.load_config import build_factorial_design, load_config
from src.methods._base_class import BaseEstimationMethod


class CountingDGP:
    calls = 0

    def __init__(self, n, k, strength, rng, **kwargs):
        self.n, self.k, self.strength, self.rng = n, k, strength, rng
        self.is_null = strength == 0

    def get_name(self):
        return "CountingDGP"

    def generate(self):
        type(self).calls += 1
        A = self.rng.random((self.n, self.n)) + self.strength
        Z = self.rng.random((self.n, self.k))
        Y = self.rng.random((self.n, 1))
        X = self.rng.random((self.n, 1))
        return {"A": A, "Z": Z, "Y": Y, "X": X}


SOLVER_CALLS = []


def counting_solver(A, k, rng, offset=0):
    SOLVER_CALLS.append((k, offset))
    return np.full((A.shape[0], k), rng.random() + offset), None


def second_solver(A, k, rng):
    return counting_solver(A, k, rng, offset=5)


class CaptureMethod(BaseEstimationMethod):
    def __init__(self, label, mutate=False, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.mutate = mutate

    def get_name(self):
        return self.label

    def fit(self, data, **kwargs):
        self._process_input(data)
        self.captured = {
            key: None if data.get(key) is None else np.array(data[key], copy=True)
            for key in ("A", "Z", "Y", "X", "estimated_Z")
        }
        self.captured["tested_Z"] = self.Zhat.copy()
        self.captured["draw"] = int(self.rng.integers(0, 2**31))
        if self.mutate:
            for value in data.values():
                if isinstance(value, np.ndarray):
                    value[...] = -999

    def get_estimated(self):
        return self.captured


class ConstructorDimensionMethod(CaptureMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = 3


class ConstructorSolverMethod(CaptureMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = second_solver


class ConstructorOracleMethod(CaptureMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_true_latent = True


class PreEmbeddedDGP(CountingDGP):
    def generate(self):
        data = super().generate()
        data["estimated_Z"] = np.full((self.n, self.k), 7.0)
        return data


class ReturnRawAMethod:
    def __init__(self, **kwargs):
        pass

    def get_name(self):
        return "raw-A"

    def fit(self, data, **kwargs):
        self.data = data

    def get_estimated(self):
        return self.data["A"]


class MutatingMetric:
    def get_name(self):
        return "mutating"

    def __call__(self, result, is_null=None):
        result[...] = 0
        return None


class WaitForCallbackMethod(CaptureMethod):
    def fit(self, data, callback_marker=None, **kwargs):
        deadline = time.monotonic() + 5
        while not Path(callback_marker).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert Path(callback_marker).exists()
        super().fit(data, **kwargs)


def capture_factory(**kwargs):
    return CaptureMethod(**kwargs)


class CaptureMetric:
    def get_name(self):
        return "capture"

    def __call__(self, result, is_null=None):
        return result


def row(label, *, strength=0.25, solver=counting_solver, k=2,
        use_true_latent=False, mutate=False, solver_kwargs=None, alpha=0.05):
    generation = {"setup": (CountingDGP, solver), "n": 6,
                  "k": 2, "strength": strength}
    result = {**generation, "method": partial(CaptureMethod, label=label, mutate=mutate),
              "use_true_latent": use_true_latent, "alpha": alpha,
              "npermutations": 3, "_generation_keys": tuple(generation)}
    if k != 2:
        result["force_k"] = k
    if solver_kwargs is not None:
        result["solver_kwargs"] = solver_kwargs
    return result


def run(design, nsim=1, seed=19, **kwargs):
    return run_simulation(nsim, design, [CaptureMetric()],
                          rng=np.random.default_rng(seed), **kwargs)


def normalized(results):
    normalized_rows = []
    for result in results:
        args = result["args"]
        capture = result["capture"]
        normalized_rows.append((
            args["method_name"], args["strength"], args.get("force_k"),
            args.get("use_true_latent"),
            tuple((key, None if capture[key] is None else
                   np.asarray(capture[key]).tobytes())
                  for key in ("A", "Z", "Y", "X", "estimated_Z", "tested_Z")),
            capture["draw"], float(result["density"]),
        ))
    return sorted(normalized_rows, key=repr)


def test_generate_and_embed_once_per_replicate_and_share_all_inputs():
    CountingDGP.calls = 0
    SOLVER_CALLS.clear()
    design = [row("first"), row("second", alpha=0.1), row("third")]
    design[2]["custom_method_option"] = "ignored by the DGP"
    results = run(design, nsim=4)
    assert CountingDGP.calls == 4
    assert SOLVER_CALLS == [(2, 0)] * 4
    assert len(results) == 12
    for replicate in range(4):
        captures = [results[replicate * 3 + index]["capture"] for index in range(3)]
        for key in ("A", "Z", "Y", "X", "estimated_Z"):
            assert all(np.array_equal(captures[0][key], item[key])
                       for item in captures[1:])
    assert all("_generation_keys" not in result["args"] and
               "data" not in result["args"] and
               "estimated_Z" not in result["args"] for result in results)


def test_mutation_isolated_and_oracle_uses_true_z():
    SOLVER_CALLS.clear()
    results = run([row("mutator", mutate=True), row("follower"),
                   row("oracle", use_true_latent=True)])
    captures = [result["capture"] for result in results]
    for key in ("A", "Z", "Y", "X"):
        np.testing.assert_array_equal(captures[0][key], captures[1][key])
        np.testing.assert_array_equal(captures[1][key], captures[2][key])
    np.testing.assert_array_equal(captures[0]["estimated_Z"],
                                  captures[1]["estimated_Z"])
    np.testing.assert_array_equal(captures[2]["tested_Z"], captures[2]["Z"])
    np.testing.assert_array_equal(captures[1]["tested_Z"],
                                  captures[1]["estimated_Z"])
    assert captures[2]["estimated_Z"] is None
    assert len(SOLVER_CALLS) == 1


def test_distinct_dgp_inputs_and_embedding_keys():
    CountingDGP.calls = 0
    SOLVER_CALLS.clear()
    design = [row("base"), row("different-strength", strength=0.8),
              row("dimension-three", k=3),
              row("solver-option", solver_kwargs={"offset": 2}),
              row("same-option", solver_kwargs={"offset": 2}),
              row("different-solver", solver=second_solver)]
    results = run(design, nsim=2)
    assert CountingDGP.calls == 4
    assert sorted(SOLVER_CALLS) == sorted(
        [(2, 0), (2, 0), (3, 0), (2, 2), (2, 5)] * 2
    )
    assert len(results) == 12
    first = [result for result in results if result["args"]["method_name"] == "base"][0]
    second = [result for result in results if result["args"]["method_name"] == "different-strength"][0]
    assert not np.array_equal(first["capture"]["A"], second["capture"]["A"])


def test_repeated_order_parallel_and_shards_are_equivalent():
    design = [row("a"), row("b", alpha=0.1), row("oracle", use_true_latent=True),
              row("other-data", strength=0.9)]
    whole = run(design, nsim=3, seed=77)
    repeated = run(design, nsim=3, seed=77)
    reordered = run(list(reversed(design)), nsim=3, seed=77)
    added_method = run(design + [row("new")], nsim=3, seed=77)
    parallel = run(design, nsim=3, seed=77, parallel=True, n_jobs=2)
    shards = [run(design, nsim=3, seed=77, shard_index=index, num_shards=3)
              for index in range(3)]
    assert normalized(whole) == normalized(repeated)
    assert normalized(whole) == normalized(reordered)
    assert normalized(whole) == normalized(
        [result for result in added_method
         if result["args"]["method_name"] != "new"]
    )
    assert normalized(whole) == normalized(parallel)
    assert normalized(whole) == normalized([item for shard in shards for item in shard])
    assert sum(shard_row_count(3, design, index, 3) for index in range(3)) == len(whole)


def test_callbacks_and_schema_match_design_rows():
    design = [row("a"), row("b")]
    seen = []
    returned = run(design, nsim=2, result_callback=seen.append)
    assert returned is None
    assert len(seen) == 4
    assert all(set(result) == {"capture", "args", "density"} for result in seen)
    assert all(result["args"]["dgp_name"] == "CountingDGP" for result in seen)
    assert [result["args"]["method_name"] for result in seen] == ["a", "b", "a", "b"]


def test_constructor_dimension_and_factory_use_actual_method_embedding_settings():
    SOLVER_CALLS.clear()
    altered = row("altered")
    altered["method"] = partial(ConstructorDimensionMethod, label="altered")
    factory = row("factory")
    factory["method"] = partial(capture_factory, label="factory")
    ordinary = row("ordinary")
    results = run([altered, factory, ordinary])
    assert SOLVER_CALLS == [(3, 0), (2, 0)]
    assert results[0]["capture"]["estimated_Z"].shape[1] == 3
    np.testing.assert_array_equal(results[1]["capture"]["estimated_Z"],
                                  results[2]["capture"]["estimated_Z"])


def test_constructor_solver_and_oracle_settings_control_embedding():
    SOLVER_CALLS.clear()
    solver_row = row("constructor-solver")
    solver_row["method"] = partial(ConstructorSolverMethod, label="constructor-solver")
    oracle_row = row("constructor-oracle")
    oracle_row["method"] = partial(ConstructorOracleMethod, label="constructor-oracle")
    results = run([solver_row, oracle_row])
    assert SOLVER_CALLS == [(2, 5)]
    np.testing.assert_array_equal(results[1]["capture"]["tested_Z"],
                                  results[1]["capture"]["Z"])


def test_dgp_supplied_embedding_is_preserved():
    SOLVER_CALLS.clear()
    supplied = row("supplied")
    supplied["setup"] = (PreEmbeddedDGP, counting_solver)
    result = run([supplied])[0]
    assert SOLVER_CALLS == []
    assert np.all(result["capture"]["estimated_Z"] == 7)


def test_density_is_measured_after_fit_and_before_metrics():
    raw = row("raw")
    raw["method"] = ReturnRawAMethod
    result = run_simulation(1, [raw], [MutatingMetric()], rng=np.random.default_rng(19))[0]
    assert result["density"] == 0


def test_generation_keys_keep_bool_and_int_distinct():
    design = [row("boolean", strength=True), row("integer", strength=1)]
    assert len(_groups(design)) == 2


def test_distinct_local_dgp_callables_do_not_share_data():
    def make_dgp(offset):
        def factory(**kwargs):
            kwargs["strength"] += offset
            return CountingDGP(**kwargs)
        return factory

    first = row("first")
    second = row("second")
    first["setup"] = (make_dgp(0), counting_solver)
    second["setup"] = (make_dgp(1), counting_solver)
    assert len(_groups([first, second])) == 2


def test_multirow_design_requires_explicit_generation_metadata():
    first, second = row("first"), row("second", solver_kwargs={"offset": 2})
    del first["_generation_keys"]
    del second["_generation_keys"]
    with pytest.raises(ValueError, match="Multirow designs require"):
        run([first, second])


@pytest.mark.parametrize("parallel", [False, True])
def test_callback_is_invoked_before_next_method_finishes(parallel, tmp_path):
    marker = tmp_path / "callback-ran"
    first, second = row("first"), row("second")
    second["method"] = partial(WaitForCallbackMethod, label="second")
    seen = []

    def callback(result):
        seen.append(result)
        if len(seen) == 1:
            marker.touch()

    run([first, second], parallel=parallel, n_jobs=2,
        method_params={"callback_marker": str(marker)}, result_callback=callback)
    assert [result["args"]["method_name"] for result in seen] == ["first", "second"]


def test_direct_data_branch_rejected_cleanly():
    with pytest.raises(ValueError, match="Direct data input is unsupported"):
        from src.helper_functions.simulation_functions import run_scenario
        run_scenario([], {**row("single"), "data": {}}, seed=1)


@pytest.mark.parametrize("config_name", [
    "config_gaussian.yaml", "config_functionals.yaml", "config_sbm.yaml",
    "config_multiness.yaml",
])
def test_existing_yaml_designs_label_generation_inputs(config_name):
    design = build_factorial_design(load_config(config_name))
    assert design
    for item in design:
        keys = set(item["_generation_keys"])
        assert {"setup", "n", "k"} <= keys
        assert not keys & {"method", "alpha", "npermutations", "df",
                           "approximation", "use_true_latent"}
    assert all(len(item["_generation_keys"]) ==
               len(set(item["_generation_keys"])) for item in design)


def test_existing_yaml_rows_execute_and_flatten_without_generated_arrays():
    import pandas as pd

    from src.helper_functions.simulation_functions import _data_key
    from src.load_config import flatten_args_columns
    from src.metrics import ReturnMetric

    design = build_factorial_design(load_config("config_gaussian.yaml"))
    first_key = _data_key(design[0])
    matching = [item for item in design if _data_key(item) == first_key]
    selected = [matching[0], matching[2]]
    for item in selected:
        item["n"] = 14
        item["npermutations"] = 2
    results = run_simulation(
        2, selected, [ReturnMetric(only_return="p-value")],
        rng=np.random.default_rng(123),
    )
    assert len(results) == 4
    frame = flatten_args_columns(pd.DataFrame(results))
    assert {"method", "dgp", "density", "rho", "solver", "n", "k"} <= set(frame)
    assert all("estimated_Z" not in args and "data" not in args
               for args in frame["args"])
