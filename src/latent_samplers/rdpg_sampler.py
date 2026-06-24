
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
        kx=None,
        rho=0,
        rdpg_distr=None,
        rdpg_params=None,
        rng=None,
        dependence_type='linear',
        noise_variance=0.1,
        **kwargs,
    ):
        self.n = n
        if kx is None:
            self.kx = k
        else:
            self.kx = kx
                
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
        Samples X and Z matrices for an RDPG ensuring that
        the inner products of any two vectors are in [0, 1].
        """
        if self.rdpg_distr == "dirichlet":
            # Default alpha is an array of ones (uniform over the simplex)
            alpha_x = self.rdpg_params.get("alpha_x", np.ones(self.kx))
            alpha_z = self.rdpg_params.get("alpha_z", np.ones(self.kz))
            # Dirichlet vectors inherently sum to 1 and are strictly non-negative.
            # Their dot products are guaranteed to be in [0, 1].
            X = self.rng.dirichlet(alpha_x, size=self.n)
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

            X = sample_positive_ball(self.kx)
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

            X = sample_trunc_norm(self.kx)
            Z = sample_trunc_norm(self.kz)

        else:
            raise ValueError(f"Unknown rdpg_distr: {self.rdpg_distr}")

        Z, X = self._induce_dependence(X, Z)

        return Z, X
    
    def _induce_dependence(self, X, Z):
        if self.rho == 0:
            self.is_null = True
            return Z, X  # No dependence to induce
        
        self.is_null = False  # Dependence is being induced
        
        if self.dependence_type == "linear":
                Z = self.rho * X + (1.0 - self.rho) * Z
        elif self.dependence_type == "quadratic":
            X2 = X**2
            norms = np.linalg.norm(X2, axis=1, keepdims=True)
            X2 = np.where(norms > 1, X2 / norms, X2)
            Z = self.rho * X2 + (1.0 - self.rho) * Z
        else:
            raise ValueError(f"Unknown dependence_type: {self.dependence_type}")
        
        return Z, X
            
    def get_name(self):
        return f"RDPG_{self.rdpg_distr}"