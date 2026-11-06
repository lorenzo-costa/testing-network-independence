"""A graph-wide mixture of independent and coupled community labels."""

from numbers import Integral
import numpy as np


class CliqueMixtureSampler:
    """Sample one regime per call, one-hot communities Z, and binary Y.

    pi is the probability of the coupled regime. B and community_labels are
    diagnostics only; neither is supplied as a tested covariate.
    """

    def __init__(self, n, k=2, ky=1, pi=0.5, rng=None, **kwargs):
        if not isinstance(n, Integral) or isinstance(n, bool) or n < 2:
            raise ValueError('n must be an integer >= 2.')
        if k != 2 or ky != 1:
            raise ValueError('CliqueMixtureSampler requires k=2 and ky=1.')
        if not np.isfinite(pi) or not 0 <= pi <= 1:
            raise ValueError('pi must be in [0, 1].')
        self.n, self.pi = int(n), float(pi)
        self.rng = np.random.default_rng(rng)
        self.is_null = self.pi == 0

    def sample_latent(self):
        self.B = int(self.rng.random() < self.pi)
        y = self.rng.binomial(1, .5, self.n)
        z = y.copy() if self.B else self.rng.binomial(1, .25, self.n)
        self.community_labels = z
        sizes = np.bincount(z, minlength=2)
        self.diagnostics = {
            'regime': self.B,
            'degenerate': bool(sizes.min() < 2 or np.unique(y).size == 1),
        }
        return np.eye(2)[z], y[:, None], None

    def get_name(self):
        return f'CliqueMixture_pi={self.pi:g}'
