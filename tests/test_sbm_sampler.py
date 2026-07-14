import importlib

import pytest

from src.latent_samplers.sbm_sampler import SBMGenerator


def test_module_imports_sbm_generator():
    module = importlib.import_module("src.latent_samplers.sbm_sampler")

    assert module.SBMGenerator is SBMGenerator


def test_unknown_assignment_mode_is_rejected():
    sampler = SBMGenerator(n=10, k=2, assignment_mode="invalid")

    with pytest.raises(ValueError, match="Unknown assignment_mode: invalid"):
        sampler._sample_community_assignment()


def test_unknown_block_probability_type_is_rejected():
    sampler = SBMGenerator(n=10, k=2, block_probs_type="invalid")

    with pytest.raises(ValueError, match="Unknown block_probs_type: invalid"):
        sampler._sample_block_probs()


@pytest.mark.parametrize("block_probs_type", ["identical", "correlated", "switched"])
def test_linked_block_probability_types_require_matching_dimensions(block_probs_type):
    sampler = SBMGenerator(
        n=10,
        k=2,
        kx=3,
        block_probs_type=block_probs_type,
    )

    with pytest.raises(ValueError, match="kx and kz must be the same"):
        sampler._sample_block_probs()

