import numpy as np
from scipy.special import expit

from .latent_samplers import LatentSampler


class GaussianNetwork(LatentSampler):
    """Weighted single-network DGP with observed node covariates."""

    def __init__(
        self,
        n,
        k,
        ky=1,
        edge_var=1,
        symmetric=True,
        self_loops=False,
        sparsity_exponent=0,
        rng=None,
        Y=None,
        Z=None,
        X=None,
        **kwargs,
    ):
        rng = rng if rng is not None else np.random.default_rng()
        super().__init__(n=n, k=k, ky=ky, rng=rng, **kwargs)
        self.edge_var = edge_var
        self.symmetric = symmetric
        self.self_loops = self_loops
        self.sparsity_exponent = sparsity_exponent
        self.Y = Y
        self.Z = Z
        self.X = X

    def __repr__(self):
        return (
            self.get_name()
            + f"(n={self.n}, k={self.k}, ky={self.ky}, edge_var={self.edge_var}, "
            f"symmetric={self.symmetric}, self_loops={self.self_loops}, "
            f"sparsity_exponent={self.sparsity_exponent})"
        )

    def get_name(self):
        return "GaussianNetwork_" + self.sampler_name

    def _get_latent_and_covariate(self):
        if (self.Z is None) != (self.Y is None):
            raise ValueError("Z and Y must either both be supplied or both be sampled.")
        if self.Z is not None:
            Z, Y, X = self.Z, self.Y, self.X
        else:
            Z, Y, X = self._sample_latent()
        Z = np.asarray(Z)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Z.shape != (self.n, self.k):
            raise ValueError(f"Z must have shape ({self.n}, {self.k}); got {Z.shape}.")
        if Y.shape != (self.n, self.ky):
            raise ValueError(f"Y must have shape ({self.n}, {self.ky}); got {Y.shape}.")
        if X is not None:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape != (self.n, 1):
                raise ValueError(f"X must have shape ({self.n}, 1); got {X.shape}.")
        self.X = X
        return Z, Y, X

    def generate(self):
        Z, Y, X = self._get_latent_and_covariate()
        expected_A = Z @ Z.T
        if self.sparsity_exponent > 0:
            expected_A *= self.n ** (-self.sparsity_exponent)

        A = self.rng.normal(loc=expected_A, scale=np.sqrt(self.edge_var))
        if not self.self_loops:
            A[np.diag_indices_from(A)] = 0
        if self.symmetric:
            A = (A + A.T) / 2
        return {"A": A, "Z": Z, "Y": Y, "X": X}


class BernoulliNetwork(GaussianNetwork):
    """Single-network Bernoulli DGP with observed node covariates."""

    def __init__(
        self,
        n,
        k,
        ky=1,
        rng=None,
        symmetric=True,
        self_loops=False,
        rdpg=False,
        sparsity_exponent=0,
        Y=None,
        Z=None,
        X=None,
        **kwargs,
    ):
        super().__init__(
            n=n,
            k=k,
            ky=ky,
            rng=rng,
            symmetric=symmetric,
            self_loops=self_loops,
            sparsity_exponent=sparsity_exponent,
            Y=Y,
            Z=Z,
            X=X,
            **kwargs,
        )
        self.rdpg = rdpg

    def get_name(self):
        return "BernoulliNetwork_" + self.sampler_name

    def __repr__(self):
        return (
            self.get_name()
            + f"(n={self.n}, k={self.k}, ky={self.ky}, rdpg={self.rdpg}, "
            f"symmetric={self.symmetric}, self_loops={self.self_loops}, "
            f"sparsity_exponent={self.sparsity_exponent})"
        )

    def generate(self):
        Z, Y, X = self._get_latent_and_covariate()
        expected_A = Z @ Z.T
        if self.sparsity_exponent > 0:
            expected_A *= np.log(self.n) ** (-self.sparsity_exponent)
        if not self.rdpg:
            expected_A = expit(expected_A)
        expected_A = np.clip(expected_A, 0, 1)

        if self.symmetric:
            triangle = self.rng.binomial(1, expected_A)
            triangle = np.tril(triangle, k=0 if self.self_loops else -1)
            A = triangle + triangle.T
            if self.self_loops:
                A[np.diag_indices_from(A)] //= 2
        else:
            A = self.rng.binomial(1, expected_A)
            if not self.self_loops:
                A[np.diag_indices_from(A)] = 0
        return {"A": A, "Z": Z, "Y": Y, "X": X}
