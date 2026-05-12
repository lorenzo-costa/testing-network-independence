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

`src/dgp.py` provides `GaussianNetwork` and `BernoulliNetwork`, both inheriting from `CopulaDGP` and `BaseSBM`. Key constructor arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `n` | `int` | Number of nodes |
| `k` | `int` | Latent space dimensionality |
| `rho` | `float` | Copula correlation parameter (0 = independent) |
| `marginals` | `str` or `dict` | Marginal distribution(s) (`'gaussian'`, `'uniform -1 1'`, `'cauchy'`, …); pass a dict with `'x'`/`'z'` keys for asymmetric marginals |
| `copula_model` | `str` | Dependency structure (see table above) |
| `edge_var` | `float` | Edge noise variance (Gaussian network only) |
| `column_covariance` | `ndarray` | Optional `k×k` covariance for the copula Gaussian factor |
| `latent_sim` | `str` | Name of a `hyppo` simulation function (e.g. `'quadratic'`, `'spiral'`) used instead of the copula path |
| `sim_kwargs` | `dict` | Extra keyword arguments forwarded to the `latent_sim` function |
| `sbm` | `bool` | Use a stochastic block model to draw latent positions |
| `dim_common` / `dim_individual` | `int` | For multi-network experiments: shared and private latent dimensions |
| `shared_latent_type` | `str` | How shared dimensions are drawn: `'gaussian'` or `'one_hot'` |
| `rdpg` | `str` | Normalisation strategy for `BernoulliNetwork` (`'max'`, `'spectral'`, `'minmax'`) |

`generate()` returns a dict with keys `A`, `B` (adjacency matrices) and `Z`, `X` (true latent positions).

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

All methods inherit from `BaseMethod` and expose `fit(data)`, `get_estimated()`, and `get_name()`. The `data` dict can contain either raw adjacency matrices (`A`, `B`) or pre-computed embeddings (`estimated_X`, `estimated_Z`).

| Class | Key parameters | Notes |
|-------|---------------|-------|
| `RVtest` | `approximation` (`'permutation'` / `'asymptotic'`), `permutation_type` (`'latent'` / `'observed'`), `npermutations`, `solver` | The asymptotic branch uses the Imhof method (`imhof.py`) to compute the p-value |
| `ObservedCVM` | `test_function` | CvM statistic on adjacency matrices; no embedding step needed |
| `LLKRatioTest` | — | Likelihood-ratio test |
| `QAP` | — | Quadratic Assignment Procedure |
| `DiffusionCorrelation` | — | Diffusion-map based correlation |
| `CanonicalCorrelationTest` | `permutation_type`, `solver` | Permutation test via canonical correlations of estimated latent positions |
| `FitIndependent` | `solver`, `k` | Not a test; fits the solver independently to each network and stores embeddings |

### Metrics (`metrics.py`)

`ComputeAll` is the recommended metric class — it computes both testing outcomes and latent-position recovery errors in a single pass.

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
