
import numpy as np

class RDPGGenerator:
    """Base class for sampling continuous latent positions for an RDPG.

    Parameters
    ----------
    n : int
        Number of samples (nodes).
    k : int
        Dimensionality of the latent space.
    rdpg_distr : str
        Distribution to sample latent positions from.
        Options: "dirichlet", "uniform_ball", "truncated_normal"
    rdpg_params : dict, optional
        Additional parameters for the chosen distribution (e.g., 'alpha' for dirichlet).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    single_network : bool
        If True, generate latent positions Z for one network and covariates/outcomes
        possiby dependent on Z.
    """

    def __init__(
        self,
        n,
        k,
        ky=1,
        rho=0,
        rdpg_distr=None,
        rdpg_params=None,
        rng=None,
        dependence_type='linear',
        noise_variance=0.1,
        **kwargs,
    ):
        self.n = n
        self.ky = ky

        self.kz = k

        self.rho = rho
        self.rdpg_distr = rdpg_distr
        self.rdpg_params = rdpg_params if rdpg_params is not None else {}
        self.rng = rng or np.random.default_rng()
        self.is_null = True
        self.dependence_type = dependence_type
        self.noise_variance = noise_variance

    def _sample_latent_rdpg(self):
        """
        Samples Y and Z matrices for an RDPG ensuring that
        the inner products of any two vectors are in [0, 1].
        """
        if self.rdpg_distr == "dirichlet":
            # Default alpha is an array of ones (uniform over the simplex)
            alpha_y = self.rdpg_params.get("alpha_y", np.ones(self.ky))
            alpha_z = self.rdpg_params.get("alpha_z", np.ones(self.kz))
            # Dirichlet vectors inherently sum to 1 and are strictly non-negative.
            # Their dot products are guaranteed to be in [0, 1].
            Y = self.rng.dirichlet(alpha_y, size=self.n)
            Z = self.rng.dirichlet(alpha_z, size=self.n)

        elif self.rdpg_distr == "uniform_ball":

            def sample_positive_ball(k):
                # sample from standard normal, normalize to sphere
                norm_samples = self.rng.normal(size=(self.n, k))
                radii = np.linalg.norm(norm_samples, axis=1, keepdims=True)
                sphere_samples = norm_samples / radii

                # Scale radially to fill the interior of the k-dimensional ball
                u = self.rng.uniform(size=(self.n, 1))
                ball_samples = sphere_samples * (u ** (1.0 / k))

                # Take absolute value to constrain to the positive orthant.
                # L2 norm <= 1 and non-negative coordinates guarantee inner products in [0, 1].
                return np.abs(ball_samples)

            Y = sample_positive_ball(self.ky)
            Z = sample_positive_ball(self.kz)

        elif self.rdpg_distr == "truncated_normal":
            loc = self.rdpg_params.get("loc", 0.0)
            scale = self.rdpg_params.get("scale", 1.0)

            def sample_trunc_norm(k):
                # Sample absolute normal
                raw = np.abs(
                    self.rng.normal(loc=loc, scale=scale, size=(self.n, k))
                )

                # Scale down only the vectors that exceed a norm of 1
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                raw = np.where(norms > 1.0, raw / norms, raw)
                return raw

            Y = sample_trunc_norm(self.ky)
            Z = sample_trunc_norm(self.kz)

        else:
            raise ValueError(f"Unknown rdpg_distr: {self.rdpg_distr}")

        Z, Y = self._induce_dependence(Y, Z)

        return Z, Y

    def _induce_dependence(self, Y, Z):
        if self.rho == 0:
            self.is_null = True
            return Z, Y  # No dependence to induce

        self.is_null = False  # Dependence is being induced

        dependent = Y
        if self.dependence_type == "linear":
            dependent = Y
        elif self.dependence_type == "quadratic":
            dependent = Y**2
            norms = np.linalg.norm(dependent, axis=1, keepdims=True)
            dependent = np.where(norms > 1, dependent / norms, dependent)
        else:
            raise ValueError(f"Unknown dependence_type: {self.dependence_type}")

        # Preserve the original coordinate-wise dependence on the dimensions
        # shared by Z and Y, leaving any remaining Z coordinates unchanged.
        common = min(self.kz, self.ky)
        Z[:, :common] = (
            self.rho * dependent[:, :common]
            + (1.0 - self.rho) * Z[:, :common]
        )

        return Z, Y

    def get_name(self):
        return f"RDPG_{self.rdpg_distr}"
