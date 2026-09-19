from .cca_test import CanonicalCorrelationTest
from .distance_correlation_test import DistanceCorrelationTest
from .QAP import MRQAP, QAP
from .rv_test import EstimateRV, RVTest

__all__ = [
    "CanonicalCorrelationTest",
    "DistanceCorrelationTest",
    "MRQAP",
    "QAP",
    "EstimateRV",
    "RVTest",
]
