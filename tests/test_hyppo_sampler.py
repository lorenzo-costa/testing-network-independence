import importlib

import numpy as np
import pytest

from src.latent_samplers.hyppo_sampler import HyppoSimSampler, SIM_REGISTRY


def test_module_imports_hyppo_sampler_and_simulations():
    module = importlib.import_module("src.latent_samplers.hyppo_sampler")

    assert module.HyppoSimSampler is HyppoSimSampler
    assert "linear" in SIM_REGISTRY
    assert callable(SIM_REGISTRY["linear"])


def test_unknown_simulation_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown sim_name 'invalid'"):
        HyppoSimSampler(n=10, k=2, sim_name="invalid")


def test_unknown_rdpg_normalization_is_rejected():
    sampler = HyppoSimSampler(n=2, k=1, sim_name="linear", make_rdpg="invalid")
    values = np.ones((2, 1))

    with pytest.raises(Exception, match="Unknown rdpg option: invalid"):
        sampler._make_rdpg(values, values)

