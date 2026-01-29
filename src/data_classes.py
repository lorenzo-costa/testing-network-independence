from dataclasses import dataclass
import numpy as np

@dataclass
class FitOutput:
    Xhat: np.ndarray
    X: np.ndarray
    def __getitem__(self, key):
        return getattr(self, key)

@dataclass
class TestOutput:
    estimated: bool
    truth: bool
    p_value: float
    def __getitem__(self, key):
        return getattr(self, key)