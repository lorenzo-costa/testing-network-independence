import numpy as np
from scipy import stats
from scipy.special import expit, ndtr

from hyppo.tools.indep_sim import (
    linear,
    exponential,
    cubic,
    quadratic,
    w_shaped,
    spiral,
    step,
    fourth_root,
    joint_normal,
    logarithmic,
    sin_four_pi,
    sin_sixteen_pi,
    square,
    diamond,
    circle,
    ellipse,
    two_parabolas,
    uncorrelated_bernoulli,
    multiplicative_noise,
    multimodal_independence,
)

# TODO:
# - extend gaussian network generation to more than two

SIM_REGISTRY = {
    "linear": linear,
    "exponential": exponential,
    "cubic": cubic,
    "quadratic": quadratic,
    "w_shaped": w_shaped,
    "spiral": spiral,
    "step": step,
    "fourth_root": fourth_root,
    "joint_normal": joint_normal,
    "logarithmic": logarithmic,
    "sin_four_pi": sin_four_pi,
    "sin_sixteen_pi": sin_sixteen_pi,
    "square": square,
    "diamond": diamond,
    "circle": circle,
    "ellipse": ellipse,
    "two_parabolas": two_parabolas,
    "uncorrelated_bernoulli": uncorrelated_bernoulli,
    "multiplicative_noise": multiplicative_noise,
    "multimodal_independence": multimodal_independence,
}
   
    
class SBMGenerator:
    """Base class for generating from a SBM
    
    Parameters
        ----------
        n : int
            Number of nodes
        k : int
            Number of communities
        num_networks : int, optional
            Number of networks, by default 2
        block_probs : _type_, optional
            _description_, by default None
        community_assignment : _type_, optional
            _description_, by default None
        assignment_mode : str, optional
            Specifies how community assignments are generated across networks. 
            Options: 
            - "random" (independent random assignment for each network) default
            - "correlated" (some nodes switch communities with some probability)
        prob_switch : float, optional
            _description_, by default 0.7
        distance_probs : _type_, optional
            _description_, by default None
    """
    
    def __init__(
        self,
        n,
        kx,
        kz,
        block_probs_type=None,
        block_probs=None,
        community_assignment=None,
        assignment_mode="random",
        assortativity=0.5,
        sparsity_bias=0.6,
        prob_switch=0.2,
        distance_probs=None,
        **kwargs,
    ):

        self.n = n
        if community_assignment is not None:
            k = community_assignment[0].shape[1]

        self.community_assignment = community_assignment
        self.kz = kz
        self.kx = kx
        self.assignment_mode = assignment_mode
        self.prob_switch = prob_switch
        self.block_probs_type = block_probs_type
        self.block_probs = block_probs
        self.distance_probs = distance_probs
        self.sparsity_bias = sparsity_bias
        self.assortativity = assortativity

    def _sample_community_assignment(self):
        assignment_z = np.zeros((self.n, self.kz))
        assignment_x = np.zeros((self.n, self.kx))

        if self.assignment_mode == "random":
            idxs_z = self.rng.integers(low=0, high=self.kz, size=self.n)
            idxs_x = self.rng.integers(low=0, high=self.kx, size=self.n)
            assignment_z[np.arange(self.n), idxs_z] = 1
            assignment_x[np.arange(self.n), idxs_x] = 1

        elif self.assignment_mode == "correlated":
            idxs_z = self.rng.integers(low=0, high=self.kz, size=self.n)
            assignment_z[np.arange(self.n), idxs_z] = 1

            switch_mask = self.rng.random(self.n) < self.prob_switch
            n_switching_nodes = np.sum(switch_mask)
            if n_switching_nodes > 0:
                shift = self.rng.integers(1, self.kx, size=n_switching_nodes)
                new_assignment = idxs_z.copy()
                new_assignment[switch_mask] = (
                    idxs_z[switch_mask] + shift
                ) % self.kz
            
            assignment_x[np.arange(self.n), new_assignment] = 1
        else:
            raise ValueError(f"Unknown assignment_mode: {self.assignment_mode}")

        self.assignment_z = assignment_z
        self.assignment_x = assignment_x
        
        return assignment_z, assignment_x

    def _generate_probability_matrix(self, k, assortativity=None):
        if assortativity is None:
            assortativity = self.assortativity

        # 1. Start with random base probabilities
        mat = self.rng.random((k, k))

        # Make symmetric (standard for undirected SBMs)
        if self.symmetric:
            mat = (mat + mat.T) / 2.0

        # 2. Create masks to separate diagonal from off-diagonal
        diag_mask = np.eye(k, dtype=bool)

        # 3. Apply Assortativity
        # Multiply diagonal by (assortativity * 2) and off-diagonal by ((1 - assortativity) * 2)
        # If assortativity=0.5, both multiply by 1.0 (no change).
        mat[diag_mask] *= assortativity * 2
        mat[~diag_mask] *= (1.0 - assortativity) * 2

        # 4. Apply Sparsity
        mat *= 1 - self.sparsity_bias

        return np.clip(mat, 0.0, 1.0)

    def _sample_block_probs(self):
        if self.block_probs_type == "random":
            probs_x = self._generate_probability_matrix(self.kx)
            probs_z = self._generate_probability_matrix(self.kz)
        elif self.block_probs_type == "identical":
            # Generate one matrix and use it for all networks
            if self.kx != self.kz:
                raise ValueError("For 'identical' block_probs_type, kx and kz must be the same.")
            probs_z = self._generate_probability_matrix(self.kx)
            probs_x = probs_z.copy()  
        elif self.block_probs_type == "correlated":
            if self.kx != self.kz:
                raise ValueError("For 'correlated' block_probs_type, kx and kz must be the same.")
            probs_z = self._generate_probability_matrix(self.kz)
            for i in range(1, self.num_networks):
                # Introduce some correlation by adding noise
                noise = self.rng.normal(loc=0.0, scale=0.1, size=probs_z.shape)
                probs_x.append(np.clip(probs_z + noise, 0.0, 1.0))
        elif self.block_probs_type == "switched":
            if self.kx != self.kz:
                raise ValueError("For 'switched' block_probs_type, kx and kz must be the same.")
            probs_z = self._generate_probability_matrix(self.kz, self.assortativity)
            probs_x = self._generate_probability_matrix(self.kx, 1 - self.assortativity)

        elif self.block_probs_type == "distance":
            raise NotImplementedError

        else:
            raise ValueError(f"Unknown block_probs_type: {self.block_probs_type}")

        self.block_probs_x = probs_x
        self.block_probs_z = probs_z

        return probs_z, probs_x

    def _sample_sbm_latent(self):
        if self.community_assignment is None:
            community_assignment_z, community_assignment_x = self._sample_community_assignment()
            self.community_assignment_z = community_assignment_z
            self.community_assignment_x = community_assignment_x
        if self.block_probs is None:
            block_probs_z, block_probs_x = self._sample_block_probs()
            self.block_probs_z = block_probs_z
            self.block_probs_x = block_probs_x

        return self.community_assignment_z, self.community_assignment_x, self.block_probs_z, self.block_probs_x


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
    def __init__(self, 
                 n,
                 k,
                 rho=0,
                 marginals="gaussian",
                 copula_model="gaussian",
                 copula_params=None,
                 column_covariance=None,
                 center_latent=True,
                 # student_df=5,
                 # weights=None,
                 # correlations=None,
                 rng=None,
                 **kwargs,
                 ):
        
        self.n = n
        self.k = k
        self.rho = rho
        self.copula_model = copula_model
        self.copula_params = copula_params if copula_params is not None else {}
        self.column_covariance = column_covariance if column_covariance is not None else np.eye(k)
        self.center_latent = center_latent
        
        self._convert_marginals(marginals)
        
        self.rng = rng or np.random.default_rng()
        
        self._validate_args_copula()
    
    def _validate_args_copula(self):
        if self.copula_model == "student_t" and "df" not in self.copula_params:
            raise ValueError("df parameter must be provided for student_t copula")
        if self.copula_model == "mixture_uniform":
            if "weights" not in self.copula_params or "correlations" not in self.copula_params:
                raise ValueError("weights and correlations must be provided for mixture_uniform copula")
            if len(self.copula_params["weights"]) != len(self.copula_params["correlations"]):
                raise ValueError("weights and correlations must have the same length for mixture_uniform copula")
            if not np.isclose(sum(self.copula_params["weights"]), 1.0):
                raise ValueError("weights must sum to 1 for mixture_uniform copula")
        
        if not self.column_covariance.shape == (self.k, self.k):
                raise ValueError(
                    f"column_covariance must be a {self.k}x{self.k} matrix."
                )    
            
    def _generate_copula_uniforms(self):
        """
        Generates Uniform(0,1) random variables (u_z, u_x)
        with the specified dependence structure.
        Returns: u_z, u_x of shape (n, k)
        """
        if self.rho is None:
            raise ValueError("rho must be passed for Copula model")

        if self.copula_model == "gaussian":
            # 1. Generate Correlated Gaussians using a k x k covariance matrix
            mean = np.zeros(self.k)
            z = self.rng.multivariate_normal(
                mean=mean, cov=self.column_covariance, size=self.n, check_valid="warn"
            )
            e = self.rng.multivariate_normal(
                mean=mean, cov=self.column_covariance, size=self.n, check_valid="warn"
            )

            if self.rho == 1:
                x = z
            elif self.rho == -1:
                x = -z
            else:
                x = self.rho * z + np.sqrt(1 - self.rho**2) * e

            # 2. Apply Gaussian CDF to get Uniforms
            u_z = ndtr(z)
            u_x = ndtr(x)

        elif self.copula_model == "student_t":
            # 1. Generate Correlated Gaussians
            df = self.copula_params["df"]
            
            mean = np.zeros(self.k)
            g_z = self.rng.multivariate_normal(
                mean=mean, cov=self.column_covariance, size=self.n
            )
            g_e = self.rng.multivariate_normal(
                mean=mean, cov=self.column_covariance, size=self.n
            )
            g_x = self.rho * g_z + np.sqrt(1 - self.rho**2) * g_e

            # 2. Generate Chi-Square variable for scaling
            w = self.rng.chisquare(df=df, size=(self.n, 1))

            # 3. Scale to create Multivariate t variables
            scale = np.sqrt(df / w)
            t_z = g_z * scale
            t_x = g_x * scale

            # 4. Apply t-distribution CDF to get Uniforms
            u_z = stats.t.cdf(t_z, df=df)
            u_x = stats.t.cdf(t_x, df=df)

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
        
        
        elif self.copula_model == 'full_clayton':
            # multivariate clayton 
            t_kendall = 2 / np.pi * np.arcsin(self.rho)
            theta = 2 * t_kendall / (1 - t_kendall)
            
            gamma_sample = self.rng.gamma(
            shape=1 / theta, scale=1.0, size=(self.n, 1)
            )

            e = self.rng.exponential(scale=1.0, size=(self.n, 2 * self.k))

            u = (1 + e / gamma_sample) ** (-1 / theta)

            u_z = u[:, :self.k]
            u_x = u[:, self.k:]
            
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

        elif self.copula_model == "mixture_uniform":
            weights = self.copula_params["weights"]
            correlations = self.copula_params["correlations"]
            
            # 1. Assign each sample (row) to a specific mixture component
            # This determines which 'rho' each row will use
            
            component_indices = self.rng.choice(
                len(weights), size=self.n, p=weights
            )

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
        self.marginal_x = parse_dist(marginals.get("x", "gaussian"))
        self.marginal_z = parse_dist(marginals.get("z", "gaussian"))
    
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


class HyppoSimSampler:
    def __init__(self, 
                 n,
                 k,
                 sim_name=None, 
                 sim_kwargs=None, 
                 center_latent=True,
                 make_rdpg=None,
                 rng=None,
                 **kwargs,
                 ):
        
        self.n = n
        self.k = k
        self.center_latent = center_latent
        self.make_rdpg = make_rdpg
        
        if (sim_name is not None) and (sim_name not in SIM_REGISTRY):
            raise ValueError(f"Unknown sim_name '{sim_name}'. Available: {sorted(SIM_REGISTRY)}")
        self.sim_name = sim_name
        self.sim_kwargs = sim_kwargs or {}
        self.rng = rng or np.random.default_rng()
    
    def _make_rdpg(self, Z, X):
        """Normalise Z and X such that inner prods are in [0, 1]"""
        if self.make_rdpg == "max":
            X = np.abs(X / np.max(X, axis=0, keepdims=True))
            Z = np.abs(Z / np.max(Z, axis=0, keepdims=True))
        elif self.make_rdpg == "spectral":
            X = X / np.sqrt(np.linalg.norm(X, ord=2))
            Z = Z / np.sqrt(np.linalg.norm(Z, ord=2))
        elif self.make_rdpg == "minmax":
            X = (X - np.min(X, axis=0, keepdims=True)) / (
                np.max(X, axis=0, keepdims=True)
                - np.min(X, axis=0, keepdims=True)
                + 1e-15
            )
            X = X / np.sqrt(X.shape[1])

            Z = (Z - np.min(Z, axis=0, keepdims=True)) / (
                np.max(Z, axis=0, keepdims=True)
                - np.min(Z, axis=0, keepdims=True)
                + 1e-15
            )
            Z = Z / np.sqrt(Z.shape[1])
            
        elif self.make_rdpg == "hypersphere":
            # Map latent values to positive orthant directions
            V_X = ndtr(X)
            V_Z = ndtr(Z)

            norm_X = np.linalg.norm(V_X, axis=1, keepdims=True)
            norm_Z = np.linalg.norm(V_Z, axis=1, keepdims=True)

            dir_X = np.where(norm_X > 0, V_X / norm_X, 0)
            dir_Z = np.where(norm_Z > 0, V_Z / norm_Z, 0)

            # Use one copula-derived coordinate for the radius
            u_x = ndtr(X[:, 0:1])
            u_z = ndtr(Z[:, 0:1])

            r_X = u_x ** (1.0 / self.k)
            r_Z = u_z ** (1.0 / self.k)

            X = r_X * dir_X
            Z = r_Z * dir_Z
        else:
            raise Exception(f"Unknown rdpg option: {self.make_rdpg}")
        
        return Z, X
    
    def _sample_latent_hyppo(self):
        """
        Use one of the simulation functions to produce (X, Z).

        The sim function is called as  sim(n, k, **sim_kwargs).
        Its two return values are treated as (X, Z):
          - first return  → X  (the 'input' latent positions)
          - second return → Z  (the 'output' latent positions)

        Shape alignment
        ---------------
        Some sims return y with shape (n, 1) instead of (n, k).
        We tile those to (n, k) so downstream code always sees (n, k).
        If the sim returns something wider than k we trim to the first k columns.
        """
        sim_fn = SIM_REGISTRY[self.latent_sim]
        
        raw_x, raw_z = sim_fn(n=self.n, p=self.k, **self.sim_kwargs)
        X = self._align_shape(np.asarray(raw_x, dtype=float))
        Z = self._align_shape(np.asarray(raw_z, dtype=float))
        
        while (not np.isfinite(X).all()) or (not np.isfinite(Z).all()):
            raw_x, raw_z = sim_fn(n=self.n, p=self.k, **self.sim_kwargs)
            X = self._align_shape(np.asarray(raw_x, dtype=float))
            Z = self._align_shape(np.asarray(raw_z, dtype=float))

        if self.center_latent:
            X = X - X.mean(axis=0)
            Z = Z - Z.mean(axis=0)
        
        if self.make_rdpg is not None:
            Z, X = self._make_rdpg(Z, X)

        return Z, X

    def _align_shape(self, arr):
        """Ensure arr has shape (n, k), tiling or trimming the column axis."""
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        _, cols = arr.shape

        if cols == self.k:
            return arr
        if cols < self.k:
            # tile: repeat columns until we reach k, then trim
            repeats = -(-self.k // cols)  # ceiling division
            arr = np.tile(arr, (1, repeats))
        return arr[:, : self.k]


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
    def __init__(self, 
                 n, 
                 k, 
                 dim_common, 
                 shared_latent_type="gaussian", 
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
    """
    def __init__(self, n, k, rdpg_distr="dirichlet", rdpg_params=None, rng=None, **kwargs):
        self.n = n
        self.k = k
        self.rdpg_distr = rdpg_distr
        self.rdpg_params = rdpg_params if rdpg_params is not None else {}
        self.rng = rng or np.random.default_rng()

    def _sample_latent_rdpg(self):
        """
        Samples X and Z matrices for an RDPG ensuring that 
        the inner products of any two vectors are in [0, 1].
        """
        if self.rdpg_distr == "dirichlet":
            # Default alpha is an array of ones (uniform over the simplex)
            alpha = self.rdpg_params.get("alpha", np.ones(self.k))
            
            # Dirichlet vectors inherently sum to 1 and are strictly non-negative.
            # Their dot products are guaranteed to be in [0, 1].
            X = self.rng.dirichlet(alpha, size=self.n)
            Z = self.rng.dirichlet(alpha, size=self.n)

        elif self.rdpg_distr == "uniform_ball":
            def sample_positive_ball():
                # sample from standard normal, normalize to sphere
                norm_samples = self.rng.normal(size=(self.n, self.k))
                radii = np.linalg.norm(norm_samples, axis=1, keepdims=True)
                sphere_samples = norm_samples / radii
                
                # Scale radially to fill the interior of the k-dimensional ball
                u = self.rng.uniform(size=(self.n, 1))
                ball_samples = sphere_samples * (u ** (1.0 / self.k))
                
                # Take absolute value to constrain to the positive orthant.
                # L2 norm <= 1 and non-negative coordinates guarantee inner products in [0, 1].
                return np.abs(ball_samples)
            
            X = sample_positive_ball()
            Z = sample_positive_ball()

        elif self.rdpg_distr == "truncated_normal":
            loc = self.rdpg_params.get("loc", 0.0)
            scale = self.rdpg_params.get("scale", 1.0)
            
            def sample_trunc_norm():
                # Sample absolute normal 
                raw = np.abs(self.rng.normal(loc=loc, scale=scale, size=(self.n, self.k)))                
                
                # Scale down only the vectors that exceed a norm of 1
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                raw = np.where(norms > 1.0, raw / norms, raw)
                return raw
            
            X = sample_trunc_norm()
            Z = sample_trunc_norm()

        else:
            raise ValueError(f"Unknown rdpg_distr: {self.rdpg_distr}")

        return Z, X

class LatentSampler(CopulaGenerator, HyppoSimSampler, OrthogonalSubspaceSampler, SBMGenerator):
    """Base class for sampling latent variables
    
    Arguments
    ----------
    n : int
        Number of samples (nodes).
    k : int
        Dimensionality of the latent space.
    copula_model : str
        Type of copula to use for generating dependence structure. If not None, function sample from Copula
    hyppo_sim : str
        Name of a simulation function from hyppo package to use for generating latent variables instead of the copula path.
        If not None function samples from specified sim function. Takes precedence over copula_model if both are provided. 
    dim_common : int
        If not None X and Z are sampled with some shared + individual structure. 
        This specifies the dimensionality of the shared subspace.
    block_probs_type : str
        If dim_common is not None, this specifies the type of shared latent structure. 
        Options: 'gaussian' (shared latent positions drawn from Gaussian) or 'one_hot' (shared latent positions are one-hot vectors indicating group membership).
    rng : np.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        n,
        k,
        copula_model=None,
        latent_sim=None,
        dim_common=None,
        block_probs_type=None,
        rdpg_distr=None,
        rng=None,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng
        self.n = n
        self.k = k
        
        CopulaGenerator.__init__(self, n=n, k=k, rng=rng, copula_model=copula_model, **kwargs)
        HyppoSimSampler.__init__(self, n=n, k=k, rng=rng, **kwargs)
        OrthogonalSubspaceSampler.__init__(self, n=n, k=k, dim_common=dim_common, rng=rng, **kwargs)
        # SBMGenerator.__init__(self, n=n, k=k, rng=rng, block_probs_type=block_probs_type, **kwargs)
        RDPGGenerator.__init__(self, n=n, k=k, rng=rng, rdpg_distr=rdpg_distr, **kwargs)
        
        self.copula_model = copula_model
        self.latent_sim = latent_sim
        self.dim_common = dim_common
        self.block_probs_type = block_probs_type
        self.rdpg_distr = rdpg_distr
    
    def _sample_latent(self):
        """Return X, Z each of shape (n, k). 
        Hierarchy of generation is:
        - if hyppo_sim is specified, use that to generate (X, Z) directly.
        - else if dim_common is specified, generate X and Z with some shared + individual structure.
        - else if block_probs_type is specified, generate X and Z with SBM structure according to the specified block probabilities.
        - else use the copula-based generation with the specified marginals and dependence structure.
        """
        Z, X = None, None
        if self.latent_sim is not None:
            Z, X = self._sample_latent_sim()
            self.sampler_name = f"HyppoSim_{self.latent_sim}"

        if self.dim_common is not None:
            if self.latent_sim is not None:
                raise Warning("Both latent_sim and dim_common specified. latent_sim will take precedence and dim_common will be ignored.")
            else:
                Z, X = self._sample_latent_orthogonal()
                self.sampler_name = f"OrthogonalSubspace_dim{self.dim_common}_{self.shared_latent_type}"

        if self.block_probs_type is not None:
            if self.latent_sim is not None or self.dim_common is not None:
                raise Warning("Multiple latent generation methods specified. block_probs_type will be ignored.")
            else:
                community_assignment_z, community_assignment_x, probs_matrix_z, probs_matrix_x = self._sample_sbm_latent()
                X = community_assignment_x @ probs_matrix_x**0.5
                Z = community_assignment_z @ probs_matrix_z**0.5
                
                self.sampler_name = f"SBM_{self.block_probs_type}"
        
        if self.rdpg_distr is not None:
            if self.latent_sim is not None or self.dim_common is not None or self.block_probs_type is not None:
                raise Warning("Multiple latent generation methods specified. rdpg_distr will be ignored.")
            else:
                Z, X = self._sample_latent_rdpg()
                self.sampler_name = f"RDPG_{self.rdpg_distr}"
        
        if self.copula_model is not None:
            if self.latent_sim is not None or self.dim_common is not None or self.block_probs_type is not None or self.rdpg_distr is not None:
                raise Warning("Multiple latent generation methods specified. copula_model will be ignored.")
            else:
                Z, X = self._sample_latent_copula()
                self.sampler_name = f"Copula_{self.copula_model}_rho{self.rho}"
        
        if Z is None or X is None:
            raise ValueError("No valid latent generation method specified. Please provide one of: latent_sim, dim_common, block_probs_type, or copula_model.")
        
        return Z, X


class GaussianNetwork(LatentSampler):
    """
    Weighted network DGP with Gaussian weights on edges.

    Allows for arbitrary marginal distributions for the latent positions Z and X,
    while maintaining a gaussian copula correlation structure.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Dimensionality of the latent space.
    edge_var : float, optional
        Variance of the edges (default is 1).
    symmetric : bool, optional
        Whether the adjacency matrix should be symmetric (default is True).
    self_loops : bool, optional
        Whether to allow self-loops (default is False).
    sparsity_exponent: float
        Controls sparsity level by multiplying expected adj matrix by n^(-sparsity_exponent).
        Higher values = sparser.
    X : np.ndarray, optional
        Pre-specified latent positions for X. If provided, these will be used instead of sampling
    Z : np.ndarray, optional
        Pre-specified latent positions for Z. If provided, these will be used instead of sampling
    
    rng : np.random.Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        n,
        k,
        edge_var=1,
        symmetric=True,
        self_loops=False,
        sparsity_exponent=0,
        rng=None,
        X=None,
        Z=None,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.default_rng()
        
        LatentSampler.__init__(self, n=n, k=k, rng=rng, **kwargs)
        
        self.edge_var = edge_var
        self.symmetric = symmetric
        self.self_loops = self_loops
        self.sparsity_exponent = sparsity_exponent
        
        self.X = X
        self.Z = Z
    
    def __repr__(self):
        return self.get_name() + f"(n={self.n}, k={self.k}, edge_var={self.edge_var}, "\
            f"symmetric={self.symmetric}, self_loops={self.self_loops}, sparsity_exponent={self.sparsity_exponent})"

    def get_name(self):
        return f"GaussianNetwork_" + self.sampler_name

    def generate(self):
        """Sample matrix and latent positions. Model definiton specifies options for:
        - latent postion type (SBM or general copula-based)
        - symmetric adjacency or not (default symmetric)
        - self loops or not (default no self loops)

        Returns
        -------
        out: dict
            dictionary with keys 'A', 'B' for adjacency matrices, 'Z', 'X' for the latent positions
        """
        if self.X is not None and self.Z is not None:
            Z, X = self.Z, self.X
        else:
            Z, X = self._sample_latent()

        expected_A = Z @ Z.T
        expected_B = X @ X.T
        
        if self.sparsity_exponent > 0:
            expected_A = expected_A * self.n**(-self.sparsity_exponent)
            expected_B = expected_B * self.n**(-self.sparsity_exponent)

        A = self.rng.normal(loc=expected_A, scale=self.edge_var)
        B = self.rng.normal(loc=expected_B, scale=self.edge_var)

        if self.self_loops is False:
            A[np.diag_indices_from(A)] = 0
            B[np.diag_indices_from(B)] = 0

        # Symmetrise
        if self.symmetric is True:
            A = (A + A.T) / 2
            B = (B + B.T) / 2

        out = {"A": A, "B": B, "Z": Z, "X": X}

        return out


class BernoulliNetwork(LatentSampler):
    """
    Network Data Generating Process using Bernoulli likelihood.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Dimensionality of the latent space.
    edge_var : float, optional
        Variance of the edges (default is 1).
    rng : np.random.Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        n,
        k,        
        rng=None,
        symmetric=True,
        self_loops=False,
        rdpg=False,
        sparsity_exponent=0,
        X=None,
        Z=None,
        **kwargs,
    ):
        LatentSampler.__init__(self, n=n, k=k, rng=rng, **kwargs)

        self.symmetric = symmetric
        self.self_loops = self_loops
        self.sparsity_exponent = sparsity_exponent
        self.rdpg = rdpg
        self.X = X
        self.Z = Z

    def get_name(self):
        return f"BernoulliNetwork_" + self.sampler_name
    
    def __repr__(self):
        return self.get_name() + f"(n={self.n}, k={self.k}, rdpg={self.rdpg}, "\
            f"symmetric={self.symmetric}, self_loops={self.self_loops}, sparsity_exponent={self.sparsity_exponent})"

    def generate(self):
        """Sample matrix and latent positions. Model definiton specifies options for:
        - latent postion type (SBM, RDPG, or general copula-based)
        - symmetric adjacency or not (default symmetric)
        - self loops or not (default no self loops)

        Returns
        -------
        out: dict
            dictionary with keys 'A', 'B' for adjacency matrices, 'Z', 'X' for the latent positions
        """
        if self.X is not None and self.Z is not None:
            X, Z = self.X, self.Z
            expected_A = Z @ Z.T
            expected_B = X @ X.T
        else:
            Z, X = self._sample_latent()
            expected_A = Z @ Z.T
            expected_B = X @ X.T
            if self.rdpg:
                # sparsity applied directly to inner product, 
                if self.sparsity_exponent > 0:
                    expected_A = expected_A * np.log(self.n)**(-self.sparsity_exponent)
                    expected_B = expected_B * np.log(self.n)**(-self.sparsity_exponent)
            else:
                # apply logit link to get probabilities
                if self.sparsity_exponent > 0:
                    expected_A = expected_A * np.log(self.n)**(-self.sparsity_exponent)
                    expected_B = expected_B * np.log(self.n)**(-self.sparsity_exponent)
                    
                expected_A = expit(expected_A)
                expected_B = expit(expected_B)

        # to be safe clip in 0, 1
        expected_A = np.clip(expected_A, 0, 1)
        expected_B = np.clip(expected_B, 0, 1)

        try:
            if self.symmetric is True:
                # generate only lower half and then sum to ensure symmetry
                A = np.tril(self.rng.binomial(1, expected_A), k=-1)
                B = np.tril(self.rng.binomial(1, expected_B), k=-1)

                A = A + A.T
                B = B + B.T
            else:
                A = self.rng.binomial(1, expected_A)
                B = self.rng.binomial(1, expected_B)
        except ValueError as e:
            # safeguard for probs not on 0, 1
            print(f"Error generating samples: {e}")
            print(f"Expected probabilities (A): {expected_A}")
            print(f"Expected probabilities (B): {expected_B}")
            raise ValueError

        if self.self_loops is False:
            A[np.diag_indices_from(A)] = 0
            B[np.diag_indices_from(B)] = 0

        out = {"A": A, "B": B, "Z": Z, "X": X}

        return out
