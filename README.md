# Network independence testing

This repository simulates a response network **Y** and **p predictor networks X**,
then tests their dependence using latent positions or adjacency matrices. The
current latent model is `Y = concatenate(X_blocks) @ B.T + epsilon`. Network
observations are Gaussian weighted or Bernoulli binary.

## Start here

Run commands from the repository root. The refactor was checked with Python
3.13.11, NumPy 2.3.5, SciPy 1.16.0, pandas 2.3.1, matplotlib 3.10.8 and Numba
0.63.1. These are tested versions, not a claim that every older version works.
NumPy must provide `Generator.spawn` for simulation and permutation streams.

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

JAX is an optional PGD backend and was not exercised in the refactor
environment. See [the refactor validation record](docs/simplification_refactor.md).

A small experiment without a configuration file:

```python
import numpy as np
from src.dgp import GaussianNetwork
from src.methods import RVTest
from src.metrics import ComputeAll
from src.solvers.weighted_network import ASE

rng = np.random.default_rng(42)
data = GaussianNetwork(n=40, p=2, d_x=2, d_y=2, snr=0.5, rng=rng).generate()
method = RVTest(solver=ASE, d_x=2, d_y=2, npermutations=19, rng=rng)
method.fit(data)
print(method.get_estimated()["p-value"])
print(ComputeAll()(method.get_estimated(), is_null=False))
```

For a larger worked example, read `run_multiple_networks_example.py`. Its defaults
run 2,400 scenarios; pass smaller parameters when exploring it.

## Reading order

| Stage | Read | Responsibility |
|---|---|---|
| Example | `run_multiple_networks_example.py` | An explicit experiment from design to results |
| Data | `src/dgp.py`, `src/latent_samplers/multiple_networks.py` | Latent blocks, coefficients, noise, and network draws |
| Method inputs | `src/methods/_network_input.py` | Validate Y/X shapes, embed each network, assemble results |
| Inference | `src/methods/_base_class.py`, concrete method files | Permutation orchestration and test statistics |
| Metrics | `src/metrics.py` | Testing outcomes and latent recovery errors |
| Configuration | `src/load_config.py`, `src/configuration/` | Resolve registries, validate fields, build ordered grids |
| Execution | `src/helper_functions/simulation_functions.py` | Scenario metadata, child seeds, workers and shards |
| Analysis | `src/analysis/` | CSV parsing, shard loading, preprocessing and generic plots |
| Figure recipes | `results/visualise_linear_model.py`, `results/visualise_linear_model_active_fraction.py` | Study-specific filters, labels, and output figures |

`A_Y` is an `(n, n)` adjacency matrix; `A_X` is an ordered list of p such
matrices. Truth is `Y` with shape `(n, d_y)` and `X`, a list of `(n, d_x)`
blocks. Methods also accept `[A_Y, A_X1, ...]` with explicit embedding dimensions.
Their result dictionaries contain `estimated_latent`, `true_latent`, `p-value`,
`reject_null`, and `test_stat`. Latent dictionaries expose `Y`, concatenated `X`,
and `X_blocks`. Missing truth is supported. Global X recovery errors compare
concatenated matrices; they are not averages of block errors.

RV and CCA cache invariant work across permutations. Distance correlation
supports dcorr and MGC; MRQAP works on adjacency matrices. Their distinct
statistical rules remain in the concrete implementations. PGD initialization and
public entry points remain in `src/solvers/MaMa_uuuuu.py`; numerical loops live in
`_pgd_backends.py`. Automatic backend selection prefers JAX, then Numba, then
NumPy. NumPy runs a fixed iteration count; Numba/JAX can stop early.

## Configured runs

```sh
python -m src.run_simulation_script --config linear_model_config.yaml
python -m src.run_sim_script_shard --config linear_model_config.yaml --shard-index 0 --num-shards 3 --n-jobs 4
```

The supplied YAMLs are research configurations and can be expensive. Inspect
`nsim`, grid sizes, and `npermutations` before running them. Multiple files can
follow `--config`; grids are concatenated in order, with execution settings,
metrics, RNG, and output options taken from the first configuration.

`src/load_config.py` retains the public DGP, solver and method registries. Add a
registered name there when extending YAML support. Builders use ordered field
sweeps: changing their order changes which scenario receives each seed.

The ordinary runner accumulates results, saves raw CSV first, then flattens
metadata. The shard runner writes bounded batches. Every shard constructs the
same global seed assignment and shuffle, then selects a disjoint strided slice.
CLI shard defaults use the corresponding Slurm array variables. Worker BLAS
threads default to one. Scenario argument dictionaries are intentionally shared
and mutated with runtime metadata; changing that ownership changes behavior.

Linear-model experiments accept `snr`, `b_active_network_fraction`, or both:

```yaml
snr: [0.1, 0.25, 0.5]
b_active_network_fraction: [0, 0.1, 0.25, 0.5, 1.0]
```

Active blocks are selected before signal calibration. If rounding fraction × p
selects no networks, the row uses `B=0`, `hypothesis=H0`, and effective `snr=0`.
A zero SNR target also gives the null. Joint sweeps retain `requested_snr`,
including separate null rows for each requested target. Fraction-only sweeps
leave coefficients uncalibrated. See `linear_model_active_fraction_config.yaml`.
With zero-mean Gaussian coefficients and positive target SNR, calibration
cancels changes in their initial overall variance.

## Results and plots

Reusable result loading, preprocessing, aggregation, and plotting live under
`src.analysis`. `preprocess_results` is the supported path for current
linear-model CSV output.

```python
from src.analysis.processing import merge_result_shards, preprocess_results
from src.analysis.plotting import plot_metric_grid
from results.visualise_linear_model import linear_model_method_label

# filenames must contain one complete shard set, not unrelated runs
raw = merge_result_shards("results", filenames)
data = preprocess_results(raw, method_labeler=linear_model_method_label)
figure, axes, summary = plot_metric_grid(
    data, value="Rejection", x="n", col="p",
    series=("method", "approximation", "asymptotic_null"),
    filters={"snr": 0}, reference_y=0.05,
)
```

Use explicit filters, facets, and grouping for every varying experiment
coordinate. For joint signal sweeps, retain both `requested_snr` and
`b_active_network_fraction`; filter by the requested SNR before plotting an
active-fraction curve.

Shard readers process bounded chunks, but merge/preparation functions collect
those chunks before concatenation: total memory is not constant. The generic
plot function returns the summary used to draw the figure. Study scripts retain
their existing labels, layouts, and filenames; use `--help` for their CLI.
The two-row type-I-error recipe needs four to six p values, even if an editable
configuration currently supplies fewer.

## Scope

The repository supports the multiple-network linear model only. The old copula,
functional, SBM, conditioning, and single-network experiment families have been
removed, along with their YAML resolution, historical tests, notebooks, and
result-processing layer. Current CSVs should be read through `src.analysis`.
