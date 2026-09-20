# TODO update this (this is AI generated)

# Network Independence Testing via Latent Position Models

A simulation framework for testing statistical independence between two networks by exploiting their shared latent position structure. The project supports both **Gaussian (weighted)** and **Bernoulli (binary)** random dot product graph models, a range of copula dependency structures, and several test statistics — including a novel observed-graph Cramér–von Mises (CvM) test that avoids explicit latent-position estimation.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Running Simulations](#running-simulations)
- [Core Components](#core-components)
  - [Data Generating Processes (`dgp.py`)](#data-generating-processes-dgppy)
  - [Solvers (`src/solvers/`)](#solvers-srcsolvers)
  - [Methods (`methods.py`)](#methods-methodspy)
  - [Metrics (`metrics.py`)](#metrics-metricspy)
  - [Helper Functions (`src/helper_functions/`)](#helper-functions-srchelper_functions)
- [Configuration (`load_config.py`)](#configuration-load_configpy)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Overview

### Supported dependency structures (copulas)

| Copula | Description |
|--------|-------------|
| `gaussian` | Gaussian copula with correlation `rho` |
| `student_t` | Student-t copula, heavier tails |
| `clayton` | Lower-tail dependent Archimedean copula |
| `rotated_clayton` | 180° rotation of the Clayton copula (upper-tail dependence) |
| `gumbel` | Upper-tail dependent Archimedean copula |
| `frank` | Symmetric Archimedean copula |
| `mixture_uniform` | Mixture of Gaussian copulas with per-component correlations |

### Supported test methods

| Method | Description |
|--------|-------------|
| `RVtest` | Permutation test using the (adjusted) RV coefficient on estimated latent positions; supports both permutation and asymptotic (Imhof) p-value approximations |
| `ObservedCVM` | CvM statistic computed directly on shared-neighbour counts — **no embedding required** |
| `LLKRatioTest` | Likelihood-ratio test |
| `QAP` | Quadratic Assignment Procedure |
| `DiffusionCorrelation` | Diffusion-map based correlation test |
| `CanonicalCorrelationTest` | Permutation test based on canonical correlations between estimated latent positions |

---

## Project Structure

```
.
├── README.md
├── Makefile
├── config.yaml                     # Experiment configuration (YAML)
├── requirements.txt
├── results/                        # Output CSVs (git-ignored)
└── src/
    ├── dgp.py                      # Data generating processes (GaussianNetwork, BernoulliNetwork)
    ├── methods.py                  # Statistical test methods
    ├── metrics.py                  # Evaluation metrics (Rejection, FrobeniusNorm, …)
    ├── load_config.py              # YAML config loader; builds factorial designs
    ├── run_simulation_script.py    # Main entry point — loads config, runs H0/H1 simulations
    ├── solvers/
    │   ├── binary_network.py       # MLE for Bernoulli RDPG (logistic regression)
    │   ├── weighted_network.py     # ASE and MLE for Gaussian RDPG
    │   ├── MaMa_uuuuu.py           # Projected gradient descent solver (pgd_fit, pgd_fit_wrapper)
    │   └── passtthrough.py         # Placeholder/pass-through solver
    └── helper_functions/
        ├── simulation_functions.py # run_simulation / run_simulation_parallel
        ├── analyse_functions.py    # aggregate_results, analyse_function
        ├── plot_functions.py       # plot_grid, plot_with_bands, plot_boxplot, …
        ├── _metrics_helper.py      # RV coefficient, CvM kernels (Numba-accelerated)
        ├── imhof.py                # Imhof method for asymptotic RV p-values
        ├── simulation_timing.py    # Timing utilities for simulation runs
        └── alternative_hp_functions.py  # Additional hypothesis-testing helpers
```

---

## Installation

**Python 3.9+ is required.**

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-name>

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

```python
import numpy as np
from functools import partial
from src.dgp import GaussianNetwork
from src.solvers.weighted_network import ASE
from src.methods import RVtest
from src.metrics import ComputeAll
from src.helper_functions.simulation_functions import run_simulation

rng = np.random.default_rng(42)

factorial_design = [
    {
        "setup": (partial(GaussianNetwork, copula_model="gaussian"), ASE),
        "method": partial(RVtest, approximation="permutation", permutation_type="latent"),
        "n": 200,
        "k": 3,
        "rho": 0.2,
        "alpha": 0.05,
        "marginals": "gaussian",
        "edge_var": 1,
        "npermutations": 100,
        "df": 3,
    }
]

results = run_simulation(
    nsim=10,
    metrics=[ComputeAll()],
    factorial_design=factorial_design,
    rng=rng,
    parallel=False,
)
print(results[0])
```

---

## Running Simulations

The recommended workflow is YAML-driven. All experiment parameters — DGPs, solvers, methods, grid values, and output paths — are specified in `config.yaml`, and a single script runs both H0 and H1 simulations.

```bash
mkdir -p results
python -m src.run_simulation_script --config config.yaml
```

Output: `results/<prefix>_<timestamp>.csv`

The script automatically detects the experiment type from the YAML structure (see [Configuration](#configuration-load_configpy)) and runs both the alternative-hypothesis sweep and the null-hypothesis baseline, then concatenates and saves the results.

---

## Core Components

### Data Generating Processes (`dgp.py`)

`src/dgp.py` provides `GaussianNetwork` and `BernoulliNetwork`. Both use
`MultipleNetworksSampler` to generate `Y = X_concat @ B.T + epsilon`, then
generate one Y network and p X networks. Key constructor arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `n` | `int` | Number of nodes |
| `p` | `int` | Number of X networks |
| `d_x`, `d_y` | `int` | Latent dimensions of each X network and Y |
| `B` | matrix, `0`, or `None` | Fixed `(d_y, p*d_x)` coefficients; `0` gives all zeros; `None` samples a new matrix per generation |
| `snr` | nonnegative `float` or `None` | Population latent signal/noise variance ratio; rescales B while keeping noise fixed; `None` disables calibration |
| `x_mean` | scalar or vector | Mean of concatenated X positions |
| `x_variance` | scalar or covariance matrix | Covariance of concatenated X; a scalar multiplies the identity |
| `eps_variance` | scalar or covariance matrix | Error covariance for Y; a scalar multiplies the identity |
| `b_mean`, `b_variance` | `float` | Mean/variance of coefficient draws before optional SNR scaling |
| `x_distribution`, `eps_distribution`, `b_distribution` | `str` | Registered latent/error/coefficient distributions |
| `edge_var` | `float` | Edge noise variance (Gaussian network only) |
| `rdpg` | `bool` | Bernoulli link: `False` uses sigmoid of latent inner products; `True` uses the inner products directly and requires valid probabilities |

`generate()` returns `A_Y`, a list `A_X`, true latent `Y`, a list `X`, and the
effective coefficient matrix `B` (after SNR scaling, when enabled).

With `snr=s`, the sampler calibrates the population ratio conditional on B:

```text
SNR = trace(B @ Sigma_X @ B.T) / trace(Sigma_epsilon)
B_effective = B_raw * sqrt(s * trace(Sigma_epsilon)
                            / trace(B_raw @ Sigma_X @ B_raw.T))
```

This is a variance ratio summed across Y dimensions, not a sample ratio or an
R-squared value. Noise covariance is unchanged. For `B=None`, every new draw
is calibrated separately; for a supplied matrix, its magnitude is rescaled
without changing its direction. `snr=0` gives zero coefficients and `Y=epsilon`.
Finite SNR requires positive noise variance; a positive target also requires
positive signal variance before scaling (`B=0, snr>0` raises an error).
Custom registered distributions must honor the configured X/error covariances
for this population interpretation to hold. The ratio concerns the latent
linear model, not Gaussian edge noise or Bernoulli edge probabilities.

```python
data = GaussianNetwork(n=200, p=5, d_x=5, d_y=5, B=None, snr=2.0).generate()
```

The notebook-safe example compares RV, CCA, and MGC at
`snr_values=(0, 0.1, 0.25, 0.5, 1)`. It uses B=0 for SNR zero and newly sampled,
calibrated B for each positive setting. With two network models and 50
repetitions per network/method/SNR combination, this gives 1,500 simulations,
without duplicating null cases. The output includes `method` and `snr` columns;
the printed rejection-rate summary groups by network, method, SNR, and
hypothesis. Pass another sequence via `snr_values` to customize the sweep.
Restart the notebook kernel before importing updated code. For the sampler and
DGP constructors, `snr=None` still disables calibration.

The same experiment can be run from the YAML-driven simulation runner:

```bash
python -m src.run_simulation_script --config linear_model_config.yaml
```

On a Slurm cluster, submit the three-shard job array with:

```bash
sbatch run_sim_shard.sbat linear_model_config.yaml
```

Each of the three array jobs requests one node with 32 CPUs. The shard runner
constructs the same deterministic global task and seed list in every job, then
executes a disjoint strided third of it. Each shard writes a separate CSV whose
name contains the Slurm array job ID and zero-based shard index; concatenate
those three CSV files to obtain the complete result table.

Edit the `n`, `p`, `d_x`, `d_y`, and `snr` lists in
`linear_model_config.yaml` to define the factorial sweep. The supplied config
runs `n` in `(50, 100, 200)`, `p` in `(5, 10, 25)`, and SNR in
`(0, 0.1, 0.25, 0.5, 1)` for Gaussian and logistic-Bernoulli networks with RV,
CCA, and MGC. It also configures simulation-level multiprocessing, one BLAS
thread per worker, and latent permutations within each method.

### Solvers (`src/solvers/`)

| Solver | File | Description |
|--------|------|-------------|
| `ASE` | `weighted_network.py` | Adjacency Spectral Embedding via truncated eigen-decomposition |
| `MLE_gaussian` | `weighted_network.py` | Shrinkage MLE for Gaussian RDPG |
| `MLE_logistic` | `binary_network.py` | Logistic-regression MLE for Bernoulli RDPG (Numba-accelerated gradient) |
| `pgd_fit` / `pgd_fit_wrapper` | `MaMa_uuuuu.py` | Projected gradient descent for binary networks |
| `placeholder_method` | `passtthrough.py` | No-op solver; returns zeros (useful for testing pipelines) |

All solvers share the signature `solver(A, k, rng, **kwargs) → (Xhat, eigenvalues)`.

### Methods (`methods.py`)

Methods expose `fit(data)` and `get_estimated()`. The multiple-network permutation
base, `RVTest`, `CanonicalCorrelationTest`, and `DistanceCorrelationTest` accept
the new DGP dictionary (`A_Y` plus a list `A_X`) or an ordered list
`[A_Y, A_X1, ..., A_Xp]`. They estimate Y using `d_y` dimensions and each X
network using `d_x` dimensions, then call the statistic as
`test_function(Yhat, Xhat)`, where `Xhat = concatenate(Xhat_blocks, axis=1)` has
shape `(n, p * d_x)`.

`permutation_type="latent"` (the default) fits all networks once and permutes
rows of `Yhat`. `permutation_type="adjacency"` permutes both axes of `A_Y` and
re-estimates Y for every permutation. The X embeddings stay fixed in both modes.
True latent Y and X can be supplied with `use_true_latent=True` in latent mode;
otherwise they are used only as reference values and to infer omitted dimensions.
`get_estimated()["estimated_latent"]` contains
`{"Y": Yhat, "X": Xhat, "X_blocks": Xhat_blocks}`;
the `"true_latent"` entry contains the corresponding true matrices and blocks,
if available. The block list preserves network order for recovery metrics.

```python
from src.methods import RVTest
from src.solvers.weighted_network import ASE

method = RVTest(
    solver=ASE, d_y=2, d_x=3,
    permutation_type="latent", npermutations=999,
    n_jobs=2, verbose=True,
)
method.fit(data)  # output of GaussianNetwork or BernoulliNetwork
```

Estimation-only methods use the same multiple-network input and output structure.

| Class | Key parameters | Notes |
|-------|---------------|-------|
| `RVTest` | `approximation` (`'permutation'` / `'asymptotic'`), `permutation_type` (`'latent'` / `'adjacency'`), `d_y`, `d_x`, `npermutations`, `solver` | The asymptotic branch uses the Imhof method (`imhof.py`) to compute the p-value |
| `ObservedCVM` | `test_function` | CvM statistic on adjacency matrices; no embedding step needed |
| `LLKRatioTest` | — | Likelihood-ratio test |
| `QAP` | — | Quadratic Assignment Procedure |
| `DiffusionCorrelation` | — | Diffusion-map based correlation |
| `CanonicalCorrelationTest` | `permutation_type`, `solver` | Permutation test via canonical correlations of estimated latent positions |
| `FitIndependent` | `solver`, `d_y`, `d_x` | Not a test; fits Y and every X network independently, stores individual X blocks, and concatenates them |

Permutation-based methods accept `n_jobs`, `batch_size`, and `verbose`.
The default `n_jobs=1` evaluates permutations serially, positive values use
that many worker processes, and `n_jobs=-1` uses every available CPU. The
process-pool chunk size targets `batch_size` work batches per worker, with a
default of 32. Set `verbose=True` to display permutation progress. Avoid
enabling permutation-level and simulation-level process parallelism at the
same time. Custom test functions and solvers must be picklable when
permutation-level parallelism is enabled. Parallel workers return completed
permutations as soon as they finish; results are restored to their original
permutation order before p-values are calculated. Permutation p-values use the
finite-sample correction `(1 + exceedances) / (npermutations + 1)`.

Multivariate AC orthant counts use the exact recursive sorted-block algorithm
from [Huang, Li and Wang, Section 5](https://arxiv.org/abs/2512.07443v2).
It supports arbitrary response dimensions, using a single-threaded, cached
Numba kernel for the 2D base case. For fixed dimension `d`, counting `q`
thresholds against `n` observations has bound `O(N * log(N)**d)`, where
`N = max(n, q)`; a coefficient with `M` neighbors has `q = n * M`.
Initial calls may incur compilation overhead. Small subproblems and inputs
requiring legacy NumPy comparison semantics use direct comparisons.
The pre-optimization helper module is preserved in
`src/test_functions/_ac_helpers_old.py`; `tests/test_orthant_counts.py` checks
exact counts, coefficients, permutation-test results, and RNG states against it.

### Metrics (`metrics.py`)

`ComputeAll` is the recommended metric class — it computes both testing outcomes and latent-position recovery errors in a single pass.

For multiple-network results it returns flat Frobenius and Procrustes error
fields for Y, each X network, and concatenated X, for example:
`RelativeFrobeniusNorm_Y`, `RelativeFrobeniusNorm_X_1`, ...,
`RelativeFrobeniusNorm_X_global` (and corresponding `ProcrustesDistance_*`
fields). The global error is evaluated on the horizontally concatenated
estimated and true X matrices, **not** the average of per-network errors.
Frobenius errors compare Gram matrices by default; existing metric formulas
are unchanged.

The individual recovery metric classes return dictionaries keyed by `Y`,
`X_1`, ..., `X_global` for named multi-network inputs. They consume the
`X_blocks` lists in method results; named inputs with X lists also work.
Legacy single-matrix and unnamed sequence inputs retain scalar/list results.
RV metrics remain similarity coefficients and MSE remains unaligned coordinate
error. `ReturnMetric("Y")` and `ReturnMetric("X")` return true Y and concatenated
true X; `ReturnMetric("estimated")` includes estimates and individual X blocks.

Unavailable true positions produce `NaN` recovery errors. Pass a known
`is_null=True` or `False` to obtain null-dependent outcome indicators; without
that label, those indicators return `NaN`, while the rejection decision remains
available. No null label is inferred from B. The simulation runner no longer
computes or returns density.

Individual metric classes: `Rejection`, `FalseRejection`, `TrueRejection`, `FalseAcceptance`, `TrueAcceptance`, `RelativeFrobeniusNorm`, `RobustRelativeProcrustesDistance`, `RVCoefficient`, `AdjustedRVCoefficient`, `MSE`.

---

## Configuration (`load_config.py`)

`load_config.py` provides a universal YAML-driven configuration system. It exposes three public functions:

```python
from src.load_config import load_config, build_factorial_design, flatten_args_columns

cfg = load_config("config.yaml")           # load and resolve config
h1, h0 = build_factorial_design(cfg)       # build factorial designs for H1 and H0
flatten_args_columns(results_df)           # post-process result DataFrames
```

The experiment type is **auto-detected** from the YAML structure — no explicit tag is required:

| Type | Detection rule | Description |
|------|---------------|-------------|
| `standard` | Default | Main copula study with H0/H1 sweep |
| `lee2019` | `setups` contains `gaussian_latent_sims` key | Latent functional-relationship study |
| `diff_marginals` | First `marginals` entry is a dict | Asymmetric per-network marginal distributions |
| `sbm` | Top-level `sbm:` key present | Stochastic block model misspecification study |
| `multiness` | `simulation` block contains `dim_common` | Multi-network common/individual latent dimensions |

Registries in `load_config.py` map YAML string names to classes — extend `DGP_REGISTRY`, `SOLVER_REGISTRY`, and `METHOD_REGISTRY` to add new components without touching the runner script.

---

## Results

Simulation outputs are written to `results/` (configurable via `config.yaml`):

| File | Contents |
|------|----------|
| `results/<prefix>_<timestamp>.csv` | Per-scenario test outcomes and metric values for both H0 and H1 runs |

The CSV columns include `n`, `k`, `rho`, `dgp`, `solver`, `method`, `marginals`, and all metric names returned by the chosen `BaseMetric` subclass.

---

## Dependencies

Core dependencies (see `requirements.txt`):

```
numpy
scipy
pandas
matplotlib
seaborn
tqdm
pyyaml
hyppo          
numba          
```
