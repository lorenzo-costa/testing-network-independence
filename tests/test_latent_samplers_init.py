import importlib


def test_package_exports_latent_sampler():
    package = importlib.import_module("src.latent_samplers")

    assert package.__all__ == [
        "LatentSampler",
        "ConditionalIndependenceCopulaSampler",
        "PostNonLinearNoiseSampler",
    ]
    assert package.LatentSampler.__name__ == "LatentSampler"
    assert (
        package.ConditionalIndependenceCopulaSampler.__name__
        == "ConditionalIndependenceCopulaSampler"
    )
