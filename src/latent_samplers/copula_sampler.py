import numpy as np
from scipy.special import ndtr
from scipy import stats


class CopulaSampler:
    """Base class for sampling from a copula-based DGP

    Parameters
    ----------
    n : int
        Number of samples (nodes).
    d_x : int
        Latent dimension of each X network.
    d_y : int, default=1
        Latent dimension of Y.
    p : int, default=1
        Number of X networks. The concatenated X vector is ordered by network
        and has dimension ``p * d_x``.
    rho : float
        Correlation parameter for the copula.
    marginals : dict or str
        Marginal distributions for the latent variables. Can be a string (e.g. 'gaussian') or a dict with 'y' and 'z' keys.
    copula_model : str
        Type of copula to use for generating dependence structure.
        Options: 'gaussian', 'student_t', 'clayton', 'rotated_clayton', 'gumbel', 'frank', 'mixture_uniform'.
    copula_params : dict
        Additional parameters for the copula model (e.g. df for student_t, weights and correlations for mixture_uniform).
    column_covariance : np.ndarray
        Covariance matrix of shape ``(p * d_x, p * d_x)`` for concatenated X.
        Off-diagonal network blocks specify dependence between X networks.
    cross_covariance : np.ndarray, optional
        X-Y covariance block of shape ``(d_y, p * d_x)``.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        n,
        d_x,
        d_y=1,
        rho=0,
        p=1,
        marginals=None,
        copula_model=None,
        copula_params=None,
        column_covariance=None,
        column_covariance_y=None,
        center_latent=True,
        cross_covariance=None,
        rng=None,
        **kwargs,
    ):
        self.n = n
        self.d_x = d_x
        self.d_y = d_y
        self.rho = rho
        self.copula_model = copula_model
        self.copula_params = copula_params if copula_params is not None else {}
        if isinstance(p, (bool, np.bool_)) or not isinstance(
            p, (int, np.integer)
        ) or p < 1:
            raise ValueError("p must be a positive integer")
        self.p = p
        self.x_dim = p * d_x
        
        self.column_covariance_x = (
            np.eye(self.x_dim)
            if column_covariance is None
            else np.asarray(column_covariance, dtype=float)
        )
        self.column_covariance_y = (
            np.eye(d_y)
            if column_covariance_y is None
            else np.asarray(column_covariance_y, dtype=float)
        )
        self.cross_covariance = (
            None
            if cross_covariance is None
            else np.asarray(cross_covariance, dtype=float)
        )

        self.center_latent = center_latent

        self._convert_marginals(marginals)

        self.rng = rng or np.random.default_rng()

        self._validate_args_copula()

        self.is_null = self._is_independence_model()

    def _is_independence_model(self):
        """Return whether the configured copula is the independence copula."""
        return self.is_independence_configuration(
            self.copula_model,
            self.rho,
            self.copula_params,
            self.cross_covariance,
        )

    @staticmethod
    def is_independence_configuration(
        copula_model, rho, copula_params, cross_covariance=None
    ):
        """Classify null configurations without drawing latent positions."""
        if cross_covariance is not None and not np.allclose(cross_covariance, 0):
            return False
        if copula_model == "gaussian":
            return rho == 0
        if copula_model == "mixture_uniform":
            correlations = np.asarray(copula_params.get("correlations", []))
            return bool(correlations.size and np.all(correlations == 0))
        # A t copula retains shared-scale dependence at rho=0. The supported
        # Archimedean copulas are likewise configured only as alternatives.
        return False

    @staticmethod
    def has_zero_covariance_configuration(
        copula_model,
        rho,
        copula_params,
        marginals,
        cross_covariance=None,
    ):
        """Verify supported copula configurations with zero cross-covariance."""
        if isinstance(marginals, str):
            gaussian_marginals = marginals == "gaussian"
        elif isinstance(marginals, dict):
            gaussian_marginals = (
                marginals.get("x") == "gaussian"
                and marginals.get("y") == "gaussian"
            )
        else:
            gaussian_marginals = False
        if not gaussian_marginals:
            return False

        if cross_covariance is not None:
            return bool(np.allclose(cross_covariance, 0))
        if copula_model == "gaussian":
            return rho == 0
        if copula_model == "mixture_uniform":
            weights = np.asarray(copula_params.get("weights", []), dtype=float)
            correlations = np.asarray(
                copula_params.get("correlations", []), dtype=float
            )
            return bool(
                weights.size
                and weights.shape == correlations.shape
                and np.isclose(weights @ correlations, 0)
            )
        return False

    def _validate_args_copula(self):
        if self.p > 1 and self.copula_model not in {
            "gaussian",
            "student_t",
            "mixture_uniform",
        }:
            raise ValueError(
                "p > 1 is supported only for gaussian, student_t, and "
                "mixture_uniform copulas"
            )
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

        if self.column_covariance_x.shape != (self.x_dim, self.x_dim):
            raise ValueError(
                "column_covariance_x must have shape "
                f"({self.x_dim}, {self.x_dim})"
            )
        if not self.column_covariance_y.shape == (self.d_y, self.d_y):
            raise ValueError(f"column_covariance_y must be a {self.d_y}x{self.d_y} matrix.")
        if self.cross_covariance is not None and self.cross_covariance.shape != (
            self.d_y,
            self.x_dim,
        ):
            raise ValueError(
                "cross_covariance must have shape "
                f"({self.d_y}, {self.x_dim})"
            )

    def _generate_gaussian(self, rho, size):
        """Helper function for generate_copula_uniforms to generate correlated Gaussian,
        with Z, Y possibly having diff dimensions.

        We define a pairing matrix R such that R_ij=1 if column i of Y is directly
        related to column j of X. Then for Sigma_X = L_X @ L_X.T and
        Sigma_Y = L_Y @ L_Y.T
        the joint cov matrix is:
        [Sigma_X, rho L_X @ R.T @ L_Y.T]
        [rho L_Y @ R @ L_X.T, Sigma_Y]
        Note cov between columns can be non-zero even if R_ij is zero through cov
        within Z and Y (i.e. column covariance)

        """

        Sigma_x = self.column_covariance_x
        Sigma_y = self.column_covariance_y

        if self.cross_covariance is None:
            Lx = np.linalg.cholesky(Sigma_x)
            Ly = np.linalg.cholesky(Sigma_y)

            def _rectangular_identity(d_y, x_dim):
                R = np.zeros((d_y, x_dim))
                m = min(d_y, x_dim)
                R[np.arange(m), np.arange(m)] = 1.0
                return R

            R = np.asarray(
                self.copula_params.get(
                    "cross_correlation_template",
                    _rectangular_identity(self.d_y, self.x_dim),
                ),
                dtype=float,
            )
            if R.shape != (self.d_y, self.x_dim):
                raise ValueError(
                    "cross_correlation_template must have shape "
                    f"({self.d_y}, {self.x_dim})"
                )

            Sigma_xy = rho * Ly @ R @ Lx.T
        else:
            # As written, this does not vary across mixture components.
            Sigma_xy = self.cross_covariance

        Sigma = np.block(
            [
                [Sigma_x, Sigma_xy.T],
                [Sigma_xy, Sigma_y],
            ]
        )

        joint = self.rng.multivariate_normal(
            mean=np.zeros(self.x_dim + self.d_y),
            cov=Sigma,
            size=size,
            check_valid="warn",
        )

        return joint

    def _generate_copula_uniforms(self):
        """
        Generates Uniform(0,1) random variables (u_z, u_y)
        with the specified dependence structure.
        Returns: u_z, u_y of shape (n, k)
        """
        if self.rho is None:
            raise ValueError("rho must be passed for Copula model")

        if self.copula_model == "gaussian":
            jj = self._generate_gaussian(rho=self.rho, size=self.n)
            x = jj[:, : self.x_dim]
            y = jj[:, self.x_dim :]

            u_x, u_y = ndtr(x), ndtr(y)
            self.is_null = self._is_independence_model()

        elif self.copula_model == "student_t":
            # Generate Gaussiana and scale by chi-sq

            jj = self._generate_gaussian(rho=self.rho, size=self.n)
            g_x = jj[:, : self.x_dim]
            g_y = jj[:, self.x_dim :]

            df = self.copula_params["df"]
            w = self.rng.chisquare(df=df, size=(self.n, 1))
            scale = np.sqrt(df / w)
            t_x = g_x * scale
            t_y = g_y * scale

            u_x = stats.t.cdf(t_x, df=df)
            u_y = stats.t.cdf(t_y, df=df)

            self.is_null = False

        elif self.copula_model == "clayton":
            # Cook & Johnson (1981) generator for Clayton
            # param is theta > 0. Larger theta = higher correlation.
            # covert rho to theta for consistency
            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 2 * t_kendall / (1 - t_kendall)

            gamma_sample = self.rng.gamma(shape=1 / theta, scale=1.0, size=(self.n, 1))

            d = self.d_x + self.d_y

            e = self.rng.exponential(scale=1.0, size=(self.n, d))

            u = (1 + e / gamma_sample) ** (-1 / theta)

            u_x = u[:, : self.d_x]
            u_y = u[:, self.d_x :]

            self.is_null = False

        elif self.copula_model == "single_clayton":
            # difference with previous fomrulation is that correlation is only
            # for columns pairs (i.e. Z_j independence X_i for i neq j)
            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 2 * t_kendall / (1 - t_kendall)

            if self.d_x != self.d_y:
                raise ValueError("single_clayton copula requires d_x == d_y")

            e_z = self.rng.exponential(scale=1.0, size=(self.n, self.d_x))
            e_y = self.rng.exponential(scale=1.0, size=(self.n, self.d_y))

            gamma_sample = self.rng.gamma(
                shape=1 / theta, scale=1.0, size=(self.n, self.d_x)
            )

            # 3. Transform
            u_x = (1 + e_z / gamma_sample) ** (-1 / theta)
            u_y = (1 + e_y / gamma_sample) ** (-1 / theta)
            self.is_null = False

        elif self.copula_model == "gumbel":
            # Generate from Gumbel–Hougaard using Marshall–Olkin-style shared-frailty
            # sampler where we draw:
            # S: from positive stable distribution with param 1/theta
            # E: iid exp(1)
            # set U_i = exp( - (E_i / S) ** alpha )

            tau = 2 / np.pi * np.arcsin(self.rho)
            theta = 1 / (1 - tau)
            alpha = 1 / theta

            d = self.d_x + self.d_y

            # genrate positive stable with the Chambers–Mallows–Stuck (CMS) method
            U = self.rng.uniform(
                low=-np.pi / 2,
                high=np.pi / 2,
                size=(self.n, 1),
            )
            W = self.rng.exponential(scale=1.0, size=(self.n, 1))

            a = np.sin(alpha * (U + np.pi / 2))
            b = np.cos(U) ** (1 / alpha)
            c = np.cos(U - alpha * (U + np.pi / 2))

            S = (a / b) * (c / W) ** ((1 - alpha) / alpha)


            E = self.rng.exponential(scale=1.0, size=(self.n, d))

            u = np.exp(-((E / S) ** alpha))

            u_x = u[:, :self.d_x]
            u_y = u[:, self.d_x:]

        elif self.copula_model == "single_gumbel":
            # difference with previous fomrulation is that correlation is only
            # for columns pairs (i.e. Z_j independence X_i for i neq j)
            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 1 / (1 - t_kendall)

            alpha = 1.0 / theta

            if self.d_x != self.d_y:
                raise ValueError("single_gumbel copula requires d_x == d_y")

            U_stab = self.rng.uniform(
                low=-np.pi / 2, high=np.pi / 2, size=(self.n, self.d_x)
            )
            W_stab = self.rng.exponential(scale=1.0, size=(self.n, self.d_x))

            a = np.sin(alpha * (U_stab + np.pi / 2))
            b = np.cos(U_stab) ** (1 / alpha)
            c = np.cos(U_stab - alpha * (U_stab + np.pi / 2))

            S = (a / b) * (c / W_stab) ** ((1 - alpha) / alpha)

            E1 = self.rng.exponential(scale=1.0, size=(self.n, self.d_x))
            E2 = self.rng.exponential(scale=1.0, size=(self.n, self.d_y))

            u_x = np.exp(-((E1 / S) ** alpha))
            u_y = np.exp(-((E2 / S) ** alpha))

            self.is_null = False

        elif self.copula_model == "frank":
            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 2 * t_kendall / (1 - t_kendall)

            # 1. Sample independent uniforms
            if self.d_x != self.d_y:
                raise ValueError("frank copula requires d_x == d_y")
            u_x = self.rng.uniform(size=(self.n, self.d_x))  # This is u
            v_raw = self.rng.uniform(
                size=(self.n, self.d_x)
            )  # This is w (conditional probability)

            # 2. Apply inverse conditional CDF to find u_y
            # Formula: u_y = -1/theta * log(1 + (v_raw * (1 - exp(-theta))) / (v_raw * (exp(-theta*u_z) - 1) - exp(-theta*u_z)))

            exp_theta = np.exp(-theta)
            exp_theta_ux = np.exp(-theta * u_x)

            numerator = v_raw * (1 - exp_theta)
            denominator = v_raw * (exp_theta_ux - 1) - exp_theta_ux
            # Usually written as: v = -1/theta * log( 1 + ... )

            # To avoid numerical instability with log, we can use log1p if needed,
            # but the standard formula is usually robust enough for moderate theta.
            arg = 1 + numerator / denominator

            # Clip arg to avoid log(negative) due to float precision issues
            arg = np.maximum(arg, 1e-10)

            u_y = -1.0 / theta * np.log(arg)

            self.is_null = False

        elif self.copula_model == "mixture_uniform":
            weights = np.asarray(self.copula_params["weights"])
            correlations = np.asarray(self.copula_params["correlations"])

            # randomly assign indices to components
            component_indices = self.rng.choice(
                len(weights),
                size=self.n,
                p=weights,
            )

            x_full = np.empty((self.n, self.x_dim))
            y_full = np.empty((self.n, self.d_y))

            for i, rho_i in enumerate(correlations):
                mask = component_indices == i
                count = mask.sum()

                if count == 0:
                    continue
                # sample gaussian conditional on component assignment
                joint = self._generate_gaussian(rho=rho_i, size=count)

                x_full[mask] = joint[:, :self.x_dim]
                y_full[mask] = joint[:, self.x_dim:]

            u_x = ndtr(x_full)
            u_y = ndtr(y_full)

            self.is_null = self._is_independence_model()

        else:
            raise NotImplementedError(f"Copula {self.copula_model} not implemented")

        return u_x, u_y

    def _convert_marginals(self, marginals):
        # 1. Normalize input into a standard format
        if marginals is None:
            raise Exception('unspecified marginals')

        if not isinstance(marginals, dict):
            marginals = {"y": marginals, "x": marginals}

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
        self.marginal_y = parse_dist(marginals["y"])
        self.marginal_x = parse_dist(marginals["x"])

    def sample_latent(self):
        u_x, u_y = self._generate_copula_uniforms()
        X = self.marginal_x.ppf(u_x)
        Y = self.marginal_y.ppf(u_y)

        # ensure no NaNs or infs from bad ppf inputs (can happen with extreme correlations and certain marginals)
        while (not np.isfinite(Y).all()) or (not np.isfinite(X).all()):
            u_x, u_y = self._generate_copula_uniforms()
            X = self.marginal_x.ppf(u_x)
            Y = self.marginal_y.ppf(u_y)

        if self.center_latent:
            Y = Y - Y.mean(axis=0)
            X = X - X.mean(axis=0)
            
        X_networks = [
            X[:, network * self.d_x : (network + 1) * self.d_x]
            for network in range(self.p)
        ]
        return {"Y": Y, "X": X_networks}

    def get_name(self):
        try:
            marginal_y_name = self.marginal_y.dist.name
            marginal_x_name = self.marginal_x.dist.name
        except AttributeError:
            marginal_y_name = str(self.marginal_y)
            marginal_x_name = str(self.marginal_x)

        return f"copula_{self.copula_model}_rho{self.rho}_marginals_{marginal_y_name}_{marginal_x_name}"
