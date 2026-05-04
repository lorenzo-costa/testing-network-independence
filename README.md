# TODO: update this


# Network Independence Testing via Latent Position Models

A simulation framework for testing statistical independence between two networks by exploiting their shared latent position structure. The project supports both **Gaussian (weighted)** and **Bernoulli (binary)** random dot product graph models, a range of copula dependency structures, and several test statistics — including a novel observed-graph Cramér–von Mises (CvM) test that avoids explicit latent-position estimation.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Running Simulations](#running-simulations)
  - [Two-stage pipeline (`run_fitting` → `run_simulation_script`)](#two-stage-pipeline)
  - [Observed-graph pipeline (`run_simulation_script_observed`)](#observed-graph-pipeline)
- [Core Components](#core-components)
  - [Data Generating Processes (`dgp.py`)](#data-generating-processes)
  - [Solvers (`src/solvers/`)](#solvers)
  - [Methods (`methods.py`)](#methods)
  - [Metrics (`metrics.py`)](#metrics)
  - [Helper Functions (`src/helper_functions/`)](#helper-functions)
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
| `gumbel` | Upper-tail dependent Archimedean copula |
| `mixture_uniform` | Mixture of two Gaussian copulas with opposing correlations |

### Supported test methods

| Method | Description |
|--------|-------------|
| `RVPermutationTest` | Permutation test using the (adjusted) RV coefficient on estimated latent positions |
| `PermutationTest` | Permutation test with a user-supplied test statistic (e.g. multivariate CvM) |
| `ObservedCVM` | CvM statistic computed directly on shared-neighbor counts — **no embedding required** |
| `LLKRatioTest` | Likelihood-ratio test |
| `QAP` | Quadratic Assignment Procedure |
| `DiffusionCorrelation` | Diffusion-map based correlation test |

---

## Project Structure

```
.
├── README.md
├── Makefile
├── requirements.txt
├── results/                        # Output CSVs and HDF5 files (git-ignored)
└── src/
    ├── dgp.py                      # Data generating processes (GaussianNetwork, BernoulliNetwork)
    ├── methods.py                  # Statistical test methods
    ├── metrics.py                  # Evaluation metrics (Rejection, FrobeniusNorm, …)
    ├── run_fitting.py              # Stage 1 — estimate latent positions, save to HDF5
    ├── run_simulation_script.py    # Stage 2 — load HDF5, run hypothesis tests
    ├── run_simulation_script_observed.py  # Single-stage observed-graph CvM pipeline
    ├── solvers/
    │   ├── binary_network.py       # MLE for Bernoulli RDPG (logistic regression)
    │   ├── weighted_network.py     # ASE and MLE for Gaussian RDPG
    │   └── MaMa_uuuuu.py           # Projected gradient descent solver (pgd_fit)
    └── helper_functions/
        ├── simulation_functions.py # run_simulation / run_simulation_parallel
        ├── analyse_functions.py    # aggregate_results, analyse_function
        ├── plot_functions.py       # plot_grid, plot_with_bands, plot_boxplot
        └── _metrics_helper.py      # RV coefficient, CvM kernels (Numba-accelerated)
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

### Optional — Numba acceleration

The CvM kernel in `_metrics_helper.py` automatically uses [Numba](https://numba.pydata.org/) JIT compilation when it is available, giving a significant speedup for large `n`. Install with:

```bash
pip install numba
```

---

## Quickstart

```python
import numpy as np
from functools import partial
from src.dgp import GaussianNetwork
from src.solvers.weighted_network import ASE
from src.methods import RVPermutationTest
from src.metrics import ComputeAll
from src.helper_functions.simulation_functions import run_simulation

rng = np.random.default_rng(42)

factorial_design = [
    {
        "setup": (partial(GaussianNetwork, copula_model="gaussian"), ASE),
        "method": partial(RVPermutationTest, permutation_type="latent"),
        "n": 200,
        "k": 3,
        "rho": 0.2,
        "alpha": 0.05,
        "marginals": "gaussian",
        "edge_var": 1,
        "approximation": "F-distr",
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

Make sure the `results/` directory exists before running any script:

```bash
mkdir -p results
```

### Two-stage pipeline

This pipeline separates latent-position estimation from hypothesis testing, which is useful when you want to evaluate multiple test statistics on the same estimated embeddings without re-running the (expensive) solvers.

**Stage 1 — fit latent positions and save to HDF5**

```bash
python -m src.run_fitting
```

Output: `results/data.h5`

**Stage 2 — load embeddings and run hypothesis tests**

```bash
python -m src.run_simulation_script
```

Output: `results/simulation_results_<timestamp>.csv`

### Observed-graph pipeline

Runs the `ObservedCVM` test directly on the adjacency matrices (no embedding step). Suitable for comparing estimation-free methods.

```bash
python -m src.run_simulation_script_observed
```

Output: `results/simulation_results_<timestamp>.csv`

---

## Core Components

### Data Generating Processes

`src/dgp.py` provides `GaussianNetwork` and `BernoulliNetwork`, both inheriting from `CopulaDGP`. Key constructor arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `n` | `int` | Number of nodes |
| `k` | `int` | Latent space dimensionality |
| `rho` | `float` | Copula correlation parameter (0 = independent) |
| `marginals` | `str` | Marginal distribution (`'gaussian'`, `'uniform -1 1'`, `'cauchy'`, …) |
| `copula_model` | `str` | Dependency structure (see table above) |
| `edge_var` | `float` | Edge noise variance (Gaussian network only) |

`generate()` returns a dict with keys `A`, `B` (adjacency matrices) and `Z`, `X` (true latent positions).

### Solvers

| Solver | File | Description |
|--------|------|-------------|
| `ASE` | `weighted_network.py` | Adjacency Spectral Embedding via truncated eigen-decomposition |
| `MLE_gaussian` | `weighted_network.py` | Shrinkage MLE for Gaussian RDPG |
| `MLE_logistic` | `binary_network.py` | Logistic-regression MLE for Bernoulli RDPG (Numba-accelerated gradient) |
| `pgd_fit` / `pgd_fit_wrapper` | `MaMa_uuuuu.py` | Projected gradient descent for binary networks |

All solvers share the signature `solver(A, k, rng, **kwargs) → (Xhat, eigenvalues)`.

### Methods

All methods inherit from `BaseMethod` and expose `fit(data)`, `get_estimated()`, and `get_name()`. The `data` dict can contain either raw adjacency matrices (`A`, `B`) or pre-computed embeddings (`estimated_X`, `estimated_Z`).

### Metrics

`ComputeAll` is the recommended metric class — it computes both testing outcomes (rejection, type-I/II error) and latent-position recovery errors (relative Frobenius norm, robust Procrustes distance) in a single pass.

Individual metric classes: `Rejection`, `FalseRejection`, `TrueRejection`, `FalseAcceptance`, `TrueAcceptance`, `RelativeFrobeniusNorm`, `RobustRelativeProcrustesDistance`.

### Helper Functions

`simulation_functions.py` — `run_simulation` dispatches to either a sequential loop or a multiprocessing pool (`run_simulation_parallel`). Set `parallel=True` to use all available CPU cores.

`_metrics_helper.py` — contains the core CvM kernel implementations. When Numba is installed the `@nb.njit(parallel=True)` variants are used automatically; otherwise a pure-NumPy fallback is used.

---

## Results

Simulation outputs are written to `results/`:

| File | Contents |
|------|----------|
| `results/data.h5` | Estimated and true latent positions (HDF5, compressed) |
| `results/simulation_results_<timestamp>.csv` | Per-scenario test outcomes and metric values |

The CSV columns include `n`, `k`, `rho`, `dgp`, `solver`, `method`, `marginals`, `approximation`, and all metric names returned by the chosen `BaseMetric` subclass.

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
h5py
numba          # optional but strongly recommended
hyppo          # for simulation registry (dgp.py)
copent         # optional, for copula mutual information
```