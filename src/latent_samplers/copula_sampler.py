
import numpy as np
from scipy.special import ndtr
from scipy import stats

class CopulaGenerator:
    """Base class for sampling from a copula-based DGP

    Parameters
    ----------
    n : int
        Number of samples (nodes).
    k : int
        Dimensionality of the latent space.
    rho : float
        Correlation parameter for the copula.
    marginals : dict or str
        Marginal distributions for the latent variables. Can be a string (e.g. 'gaussian') or a dict with 'x' and 'z' keys.
    copula_model : str
        Type of copula to use for generating dependence structure.
        Options: 'gaussian', 'student_t', 'clayton', 'rotated_clayton', 'gumbel', 'frank', 'mixture_uniform'.
    copula_params : dict
        Additional parameters for the copula model (e.g. df for student_t, weights and correlations for mixture_uniform).
    column_covariance : np.ndarray
        Covariance matrix for the columns of the latent variables. Used in Guassian, Student-t,
        and mixture_uniform copulas to induce column-wise dependence.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        n,
        k,
        kx=None,
        rho=0,
        marginals=None,
        copula_model=None,
        copula_params=None,
        column_covariance=None,
        column_covariance_x=None,
        center_latent=True,
        cross_covariance=None,
        rng=None,
        **kwargs,
    ):
        self.n = n
        if kx is None:
            kx = k
            
        self.k = k
        self.kx = kx
        self.rho = rho
        self.copula_model = copula_model
        self.copula_params = copula_params if copula_params is not None else {}
        
        if column_covariance_x is None:
            column_covariance_x = column_covariance
        
        self.column_covariance_z = (
            np.eye(k) if column_covariance is None else column_covariance
        )
        self.column_covariance_x = (
            np.eye(kx) if column_covariance_x is None else column_covariance_x
        )
            
        self.center_latent = center_latent

        self._convert_marginals(marginals)

        self.rng = rng or np.random.default_rng()

        self._validate_args_copula()

        self.is_null = True

    def _validate_args_copula(self):
        if self.copula_model == "student_t" and "df" not in self.copula_params:
            raise ValueError("df parameter must be provided for student_t copula")
        if self.copula_model == "mixture_uniform":
            if (
                "weights" not in self.copula_params
                or "correlations" not in self.copula_params
            ):
                raise ValueError(
                    "weights and correlations must be provided for mixture_uniform copula"
                )
            if len(self.copula_params["weights"]) != len(
                self.copula_params["correlations"]
            ):
                raise ValueError(
                    "weights and correlations must have the same length for mixture_uniform copula"
                )
            if not np.isclose(sum(self.copula_params["weights"]), 1.0):
                raise ValueError("weights must sum to 1 for mixture_uniform copula")

        if not self.column_covariance_z.shape == (self.kz, self.kz):
            raise ValueError(f"column_covariance_z must be a {self.kz}x{self.kz} matrix.")
        if not self.column_covariance_x.shape == (self.kx, self.kx):
            raise ValueError(f"column_covariance_x must be a {self.kx}x{self.kx} matrix.")
    
    def _generate_gaussian(self):
        """Helper function for generate_copula_uniforms to generate correlated Gaussian,
        with Z, X possibly having diff dimensions.
        
        We define a pairing matrix R such that R_ij=1 if column i of Z is directly 
        related to column j of X. Then for Sigma_Z = L_Z @ L_Z.T and Sigma_X = L_X @ L_X.T
        the joint cov matrix is:
        [Sigma_Z, rho L_Z @ R.T @ L_X.T]
        [rho L_X @ R.T @ L_Z.T, Sigma_X]
        Note cov between columns can be non-zero even if R_ij is zero through cov
        within Z and X (i.e. column covariance)
        
        """
        
        Sigma_z = self.column_covariance_z
        Sigma_x = self.column_covariance_x

        if self.cross_covariance is None:
            def _rectangular_identity(kz, kx):
                R = np.zeros((kz, kx))
                m = min(kz, kx)
                R[np.arange(m), np.arange(m)] = 1.0
                return R
            
            Lz = np.linalg.cholesky(Sigma_z)
            Lx = np.linalg.cholesky(Sigma_x)

            R = _rectangular_identity(self.kz, self.kx)

            Sigma_zx = self.rho * Lz @ R @ Lx.T
        else:
            Sigma_zx = self.cross_covariance

        Sigma = np.block([
            [Sigma_z,    Sigma_zx],
            [Sigma_zx.T, Sigma_x ],
        ])

        joint = self.rng.multivariate_normal(
            mean=np.zeros(self.kz + self.kx),
            cov=Sigma,
            size=self.n,
            check_valid="warn",
        )

        return joint

    def _generate_copula_uniforms(self):
        """
        Generates Uniform(0,1) random variables (u_z, u_x)
        with the specified dependence structure.
        Returns: u_z, u_x of shape (n, k)
        """
        if self.rho is None:
            raise ValueError("rho must be passed for Copula model")

        if self.copula_model == "gaussian":
            jj = self._generate_gaussian()
            z = jj[:, : self.kz]
            x = jj[:, self.kz :]
            
            u_z, u_x = ndtr(z), ndtr(x)
            self.is_null = (self.rho == 0)

        elif self.copula_model == "student_t":
            # Generate Gaussiana and scale by chi-sq
            
            jj = self._generate_gaussian()
            g_z = jj[:, : self.kz]
            g_x = jj[:, self.kz :]

            df = self.copula_params["df"]
            w = self.rng.chisquare(df=df, size=(self.n, 1))
            scale = np.sqrt(df / w)
            t_z = g_z * scale
            t_x = g_x * scale

            u_z = stats.t.cdf(t_z, df=df)
            u_x = stats.t.cdf(t_x, df=df)

            self.is_null = False

        elif self.copula_model == "clayton":
            # Cook & Johnson (1981) generator for Clayton
            # param is theta > 0. Larger theta = higher correlation.
            # covert rho to theta for consistency

            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 2 * t_kendall / (1 - t_kendall)

            # Generate Exponentials
            e_z = self.rng.exponential(scale=1.0, size=(self.n, self.k))
            e_x = self.rng.exponential(scale=1.0, size=(self.n, self.k))

            # Generate Gamma
            # Shape (n, 1) so dependence is tied within the pair generation
            gamma_sample = self.rng.gamma(
                shape=1 / theta, scale=1.0, size=(self.n, self.k)
            )

            # 3. Transform
            u_z = (1 + e_z / gamma_sample) ** (-1 / theta)
            u_x = (1 + e_x / gamma_sample) ** (-1 / theta)
            self.is_null = False

        elif self.copula_model == "full_clayton":
            # multivariate clayton
            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 2 * t_kendall / (1 - t_kendall)

            gamma_sample = self.rng.gamma(shape=1 / theta, scale=1.0, size=(self.n, 1))

            e = self.rng.exponential(scale=1.0, size=(self.n, 2 * self.k))

            u = (1 + e / gamma_sample) ** (-1 / theta)

            u_z = u[:, : self.k]
            u_x = u[:, self.k :]

            self.is_null = False

        elif self.copula_model == "rotated_clayton":
            # Generate standard Clayton
            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 2 * t_kendall / (1 - t_kendall)

            e_z = self.rng.exponential(scale=1.0, size=(self.n, self.k))
            e_x = self.rng.exponential(scale=1.0, size=(self.n, self.k))
            gamma_sample = self.rng.gamma(
                shape=1 / theta, scale=1.0, size=(self.n, self.k)
            )

            u_z_raw = (1 + e_z / gamma_sample) ** (-1 / theta)
            u_x_raw = (1 + e_x / gamma_sample) ** (-1 / theta)

            # 180 degree flip
            u_z = 1.0 - u_z_raw
            u_x = 1.0 - u_x_raw

            self.is_null = False

        elif self.copula_model == "gumbel":
            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 1 / (1 - t_kendall)

            alpha = 1.0 / theta

            # 1. Simulate Positive Stable Random Variables S ~ St(alpha, 1, ...)
            # Using Chambers-Mallows-Stuck method
            U_stab = self.rng.uniform(
                low=-np.pi / 2, high=np.pi / 2, size=(self.n, self.k)
            )
            W_stab = self.rng.exponential(scale=1.0, size=(self.n, self.k))

            # Intermediate terms for stable generator
            a = np.sin(alpha * (U_stab + np.pi / 2))
            b = np.cos(U_stab) ** (1 / alpha)
            c = np.cos(U_stab - alpha * (U_stab + np.pi / 2))

            S = (a / b) * (c / W_stab) ** ((1 - alpha) / alpha)

            # 2. Generate Exponentials
            E1 = self.rng.exponential(scale=1.0, size=(self.n, self.k))
            E2 = self.rng.exponential(scale=1.0, size=(self.n, self.k))

            # 3. Transform to Uniforms
            # Formula: u = exp( - (E / S)^alpha )
            u_z = np.exp(-((E1 / S) ** alpha))
            u_x = np.exp(-((E2 / S) ** alpha))

            self.is_null = False

        elif self.copula_model == "frank":
            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 2 * t_kendall / (1 - t_kendall)

            # 1. Sample independent uniforms
            u_z = self.rng.uniform(size=(self.n, self.k))  # This is u
            v_raw = self.rng.uniform(
                size=(self.n, self.k)
            )  # This is w (conditional probability)

            # 2. Apply inverse conditional CDF to find u_x
            # Formula: u_x = -1/theta * log(1 + (v_raw * (1 - exp(-theta))) / (v_raw * (exp(-theta*u_z) - 1) - exp(-theta*u_z)))

            exp_theta = np.exp(-theta)
            exp_theta_uz = np.exp(-theta * u_z)

            numerator = v_raw * (1 - exp_theta)
            denominator = v_raw * (exp_theta_uz - 1) - exp_theta_uz
            # Usually written as: v = -1/theta * log( 1 + ... )

            # To avoid numerical instability with log, we can use log1p if needed,
            # but the standard formula is usually robust enough for moderate theta.
            arg = 1 + numerator / denominator

            # Clip arg to avoid log(negative) due to float precision issues
            arg = np.maximum(arg, 1e-10)

            u_x = -1.0 / theta * np.log(arg)

            self.is_null = False

        elif self.copula_model == "mixture_uniform":
            weights = self.copula_params["weights"]
            correlations = self.copula_params["correlations"]

            # 1. Assign each sample (row) to a specific mixture component
            # This determines which 'rho' each row will use

            component_indices = self.rng.choice(len(weights), size=self.n, p=weights)

            # Initialize the full (n, k) latent Gaussian arrays
            z_full = np.zeros((self.n, self.k))
            x_full = np.zeros((self.n, self.k))

            mean = np.zeros(self.k)

            # 2. Generate data for each mixture component
            for i, rho in enumerate(correlations):
                # Find which of the 'n' rows belong to this mixture component
                mask = component_indices == i
                count = np.sum(mask)

                if count > 0:
                    # Generate Correlated Gaussians for these specific rows
                    # using the shared k x k column covariance
                    z = self.rng.multivariate_normal(
                        mean=mean, cov=self.column_covariance, size=count
                    )
                    e = self.rng.multivariate_normal(
                        mean=mean, cov=self.column_covariance, size=count
                    )

                    # Link Z and X using this component's specific rho
                    if rho == 1:
                        x = z
                    elif rho == -1:
                        x = -z
                    else:
                        x = rho * z + np.sqrt(1 - rho**2) * e

                    # Place the generated rows back into the full arrays
                    z_full[mask] = z
                    x_full[mask] = x

            # 3. Apply Gaussian CDF to the completed arrays to get Uniform margins
            u_z = ndtr(z_full)
            u_x = ndtr(x_full)

            self.is_null = False

        else:
            raise NotImplementedError(f"Copula {self.copula_model} not implemented")

        return u_z, u_x

    def _convert_marginals(self, marginals):
        # 1. Normalize input into a standard format
        if not isinstance(marginals, dict):
            marginals = {"x": marginals, "z": marginals}

        # 2. Define a helper to parse a single string/distribution
        def parse_dist(dist_str):
            parts = dist_str.split()
            name = parts[0]
            args = [float(p) for p in parts[1:]]  # Convert params to floats

            # Dispatch table: maps name to (scipy_func, arg_names)
            registry = {
                "gaussian": (stats.norm, []),
                "exponential": (stats.expon, []),
                "uniform": (stats.uniform, []),
                "t": (stats.t, ["df"]),
                "chi": (stats.chi2, ["df"]),
                "chi2": (stats.chi2, ["df"]),
                "beta": (stats.beta, []),
                "gamma": (stats.gamma, ["a", "scale"]),  # Special handling for scale
                "lognormal": (stats.lognorm, ["s"]),
                "cauchy": (stats.cauchy, ["loc", "scale"]),
                "dirichlet": (stats.dirichlet, ["alpha"]),
            }

            if name not in registry:
                raise ValueError(f"Unknown distribution: {name}")

            func, arg_keys = registry[name]

            # Handle simple positional distributions vs keyword ones
            if not args:
                return func
            if name == "uniform" and len(args) == 2:
                a, b = args
                return func(loc=a, scale=b - a)  # scale is the width (max - min)
            if name == "gamma" and len(args) == 2:
                return func(a=args[0], scale=args[1])
            if name == "cauchy" and len(args) == 2:
                return func(loc=args[0], scale=args[1])

            # Map args to keys if provided, otherwise pass as positional
            kwargs = {k: v for k, v in zip(arg_keys, args) if k}
            return func(**kwargs) if kwargs else func(*args)

        # 3. Apply to both variables
        self.marginal_x = parse_dist(marginals["x"])
        self.marginal_z = parse_dist(marginals["z"])

    def _sample_latent_copula(self):
        u_z, u_x = self._generate_copula_uniforms()
        Z = self.marginal_z.ppf(u_z)
        X = self.marginal_x.ppf(u_x)

        # ensure no NaNs or infs from bad ppf inputs (can happen with extreme correlations and certain marginals)
        while (not np.isfinite(X).all()) or (not np.isfinite(Z).all()):
            u_z, u_x = self._generate_copula_uniforms()
            Z = self.marginal_z.ppf(u_z)
            X = self.marginal_x.ppf(u_x)

        if self.center_latent:
            X = X - X.mean(axis=0)
            Z = Z - Z.mean(axis=0)
        return Z, X

    def get_name(self):
        try:
            marginal_x_name = self.marginal_x.dist.name
            marginal_z_name = self.marginal_z.dist.name
        except AttributeError:
            marginal_x_name = str(self.marginal_x)
            marginal_z_name = str(self.marginal_z)

        return f"copula_{self.copula_model}_rho{self.rho}_marginals_{marginal_x_name}_{marginal_z_name}"

