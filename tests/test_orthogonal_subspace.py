import importlib

import pytest

from src.latent_samplers.orthogonal_subspace import OrthogonalSubspaceSampler


def test_module_imports_orthogonal_subspace_sampler():
    module = importlib.import_module("src.latent_samplers.orthogonal_subspace")

    assert module.OrthogonalSubspaceSampler is OrthogonalSubspaceSampler


def test_common_dimension_cannot_exceed_total_dimension():
    sampler = OrthogonalSubspaceSampler(
        n=10,
        k=2,
        dim_common=3,
        shared_latent_type="gaussian",
    )

    with pytest.raises(ValueError, match="dim_common must not exceed"):
        sampler._sample_latent_orthogonal()


def test_unknown_shared_latent_type_is_rejected():
    sampler = OrthogonalSubspaceSampler(
        n=10,
        k=2,
        dim_common=1,
        shared_latent_type="invalid",
    )

    with pytest.raises(ValueError, match="Unknown shared_latent_type: invalid"):
        sampler._sample_latent_orthogonal()
