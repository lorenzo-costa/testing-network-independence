import numpy as np

class OrthogonalSubspaceSampler:
    """Sampler for generating X and Z with some shared + individual structure.
    Matrices are generated to be orthogonal to each other to ensure identifiability.

    Parameters
    ----------
    n : int
        Number of samples (nodes).
    k : int
        Dimensionality of the latent space.
    dim_common : int
        If not None X and Z are sampled with some shared + individual structure.
        This specifies the dimensionality of the shared subspace.
    shared_latent_type : str
        If dim_common is not None, this specifies the type of shared latent structure.
        Options: 'gaussian' (shared latent positions drawn from Gaussian) or 'one_hot' (shared latent positions are one-hot vectors indicating group membership).
    center_latent : bool
        Whether to center the latent variables to have mean zero after generation.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        n,
        k,
        dim_common,
        shared_latent_type=None,
        center_latent=True,
        rng=None,
        **kwargs,
    ):
        self.n = n
        self.k = k
        self.dim_common = dim_common
        self.shared_latent_type = shared_latent_type
        self.center_latent = center_latent
        self.rng = rng or np.random.default_rng()

        self.is_null = False

    def _sample_latent_orthogonal(self):
        """Sample X and Z with some shared + individual structure. Matrices are
        generated to be orthogonal to each other to ensure identifiability."""

        if self.dim_common > self.k:
            raise ValueError("dim_common must be specified less than k.")

        dim_common = self.dim_common
        dim_individual = self.k - dim_common

        X = np.zeros((self.n, self.k))
        Z = np.zeros((self.n, self.k))

        if self.shared_latent_type == "gaussian":
            U_gauss = np.random.randn(
                self.n, dim_common
            )  # Shared latent positions from Gaussian
            V_x_gauss = np.random.randn(
                self.n, dim_individual
            )  # X-specific latent positions
            V_z_gauss = np.random.randn(
                self.n, dim_individual
            )  # Z-specific latent positions

            U, _ = np.linalg.qr(U_gauss)  # Orthonormalize U
            V_x, _ = np.linalg.qr(V_x_gauss)  # Orthonormalize V_x
            V_z, _ = np.linalg.qr(V_z_gauss)  # Orthonormalize V_z

            X = np.hstack((U, V_x))
            Z = np.hstack((U, V_z))

        elif self.shared_latent_type == "one_hot":
            idxs = np.random.randint(0, dim_common, size=self.n)
            C = np.zeros((self.n, dim_common))
            C[np.arange(self.n), idxs] = 1.0

            V = np.random.randn(self.n, dim_individual)
            W = np.random.randn(self.n, dim_individual)

            X = np.concatenate([C, V], axis=1)
            Z = np.concatenate([C, W], axis=1)
        else:
            raise ValueError(f"Unknown shared_latent_type: {self.shared_latent_type}")

        return Z, X

    def get_name(self):
        return f"OrthogonalSubspace_{self.shared_latent_type}"
