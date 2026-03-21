# ==============================================================================
#  Makefile — Network Independence Testing Simulation Framework
# ==============================================================================
#
#  Key targets:
#    make install          — install Python dependencies
#    make results          — run full two-stage pipeline (fit → test)
#    make fit              — Stage 1: estimate latent positions → results/data.h5
#    make simulate         — Stage 2: hypothesis tests on stored embeddings
#    make simulate-observed— run observed-graph CvM pipeline (single stage)
#    make clean            — remove all generated result files
#    make clean-results    — remove only CSV result files
#    make help             — print this message
#
# ==============================================================================
 
PYTHON      ?= python
RESULTS_DIR  = results
 
# ── Environment setup ─────────────────────────────────────────────────────────
 
.PHONY: install
install:  ## Install all Python dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
 
.PHONY: install-numba
install-numba:  ## Install optional Numba JIT dependency (recommended for large n)
	$(PYTHON) -m pip install numba
 
# ── Results directory ─────────────────────────────────────────────────────────
 
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)
 
# ── Two-stage pipeline ────────────────────────────────────────────────────────
 
.PHONY: fit
fit: $(RESULTS_DIR)  ## Stage 1 — estimate latent positions and save to results/data.h5
	$(PYTHON) -m src.run_fitting
 
.PHONY: simulate
simulate: $(RESULTS_DIR)/data.h5  ## Stage 2 — run hypothesis tests on stored embeddings
	$(PYTHON) -m src.run_simulation_script
 
# Convenience target: run both stages in sequence
.PHONY: results
results: fit simulate  ## Run the full two-stage pipeline (fit then simulate)
 
# Guard: abort Stage 2 if Stage 1 has not been run yet
$(RESULTS_DIR)/data.h5:
	@echo ""
	@echo "  ERROR: $(RESULTS_DIR)/data.h5 not found."
	@echo "  Run 'make fit' first to generate the HDF5 embeddings file."
	@echo ""
	@exit 1
 
# ── Observed-graph pipeline ───────────────────────────────────────────────────
 
.PHONY: simulate-observed
simulate-observed: $(RESULTS_DIR)  ## Run the observed-graph CvM pipeline (no embedding step)
	$(PYTHON) -m src.run_simulation_script_observed
 
# ── Housekeeping ──────────────────────────────────────────────────────────────
 
.PHONY: clean-results
clean-results:  ## Remove only CSV simulation result files (keep HDF5)
	find $(RESULTS_DIR) -name "simulation_results_*.csv" -delete
	@echo "Removed simulation CSVs from $(RESULTS_DIR)/"
 
.PHONY: clean
clean:  ## Remove all generated files (CSVs and HDF5)
	rm -rf $(RESULTS_DIR)
	@echo "Removed $(RESULTS_DIR)/"
 
# ── Help ──────────────────────────────────────────────────────────────────────
 
.PHONY: help
help:  ## Print available targets
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-25s %s\n", $$1, $$2}'
	@echo ""
 
.DEFAULT_GOAL := help