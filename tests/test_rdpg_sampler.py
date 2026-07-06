import importlib

import numpy as np
import pytest

from src.latent_samplers.rdpg_sampler import RDPGGenerator


def test_module_imports_rdpg_generator():
    module = importlib.import_module("src.latent_samplers.rdpg_sampler")

    assert module.RDPGGenerator is RDPGGenerator


def test_unknown_rdpg_distribution_is_rejected():
    sampler = RDPGGenerator(n=10, k=2, rdpg_distr="invalid")

    with pytest.raises(ValueError, match="Unknown rdpg_distr: invalid"):
        sampler._sample_latent_rdpg()


def test_unknown_dependence_type_is_rejected_when_dependence_is_requested():
    sampler = RDPGGenerator(
        n=3,
        k=2,
        rho=0.5,
        rdpg_distr="dirichlet",
        dependence_type="invalid",
    )
    values = np.full((3, 2), 0.5)

    with pytest.raises(ValueError, match="Unknown dependence_type: invalid"):
        sampler._induce_dependence(values, values)

