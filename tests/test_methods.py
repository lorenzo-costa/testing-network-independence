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
