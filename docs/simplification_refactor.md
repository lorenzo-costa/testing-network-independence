# Simplification refactor

The repository now has one supported experiment path:

```
linear-model YAML → configuration grid → simulation → methods/metrics → CSV → analysis
```

`src/load_config.py` accepts only `experiment_type: linear_model`. It resolves
the two network types, supported solvers, and methods into an ordered factorial
grid. `src/latent_samplers/` contains only `MultipleNetworksSampler`, which
generates the predictor blocks and response latent positions used by the active
data-generating process.

`src/analysis/` owns shard I/O, parsing current runner output, preprocessing,
aggregation, and generic plotting. `results/visualise_linear_model.py` contains
the linear-model figure recipes. Older experiment builders, samplers, tests,
notebooks, figure recipes, and result-processing compatibility code were
removed rather than kept as unsupported branches.

The refactor keeps public method and solver locations, the existing RNG order,
parallel shard behavior, numerical backends, and CSV metadata flow for the
linear-model path. Validation: `python -m pytest -q` passes 645 current-path
tests.
