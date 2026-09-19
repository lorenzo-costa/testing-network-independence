import importlib

import src.methods as methods


def test_methods_package_exports_public_components():
    package = importlib.import_module("src.methods")
    expected_exports = {
        "CanonicalCorrelationTest",
        "DistanceCorrelationTest",
        "MRQAP",
        "QAP",
        "EstimateRV",
        "RVTest",
    }

    assert set(package.__all__) == expected_exports
    for name in expected_exports:
        assert getattr(package, name) is getattr(methods, name)


def test_config_registry_does_not_reference_deleted_ac_method():
    from src.load_config import METHOD_REGISTRY

    assert "MultivariateACTest" not in METHOD_REGISTRY
    exported_methods = {getattr(methods, name) for name in methods.__all__}
    assert set(METHOD_REGISTRY.values()) <= exported_methods
