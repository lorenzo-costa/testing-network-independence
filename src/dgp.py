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


class BaseDPG:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def generate(self):
        raise NotImplementedError("Subclasses should implement this!")

    def name(self):
        raise NotImplementedError("Subclasses should implement this!")


class CopulaDGP:
    """Base class for Copula Data Generating Processes."""

    def __init__(
        self,
        n,
        k,
        rho,
        marginals="gaussian",
        rng=None,
        copula_model="gaussian",
        df=5,
        weights=None,
        correlations=None,
        center_latent=True,
        latent_sim=None,  # NEW: name of a sim function, e.g. "quadratic"
        sim_kwargs=None,  # NEW: extra kwargs forwarded to that function
        column_covariance=None,
        rdpg_distribution=None,
        dim_common=None,
        dim_individual=None,
        shared_latent_type=None,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng
        self.n = n
        self.k = k
        self.rho = rho

        if column_covariance is not None:
            if not column_covariance.shape == (self.k, self.k):
                raise ValueError(
                    f"column_covariance must be a {self.k}x{self.k} matrix."
                )
            self.column_covariance = column_covariance
        else:
            self.column_covariance = np.eye(self.k)

        self.marginals = marginals
        self.copula_model = copula_model
        self.df = df
        self.weights = weights
        self.correlations = correlations
        self.center_latent = center_latent
        self.rdpp_distribution = rdpg_distribution

        # ── sim path ──────────────────────────────────────────────────────────
        if latent_sim is not None and latent_sim not in SIM_REGISTRY:
            raise ValueError(
                f"Unknown latent_sim '{latent_sim}'. Available: {sorted(SIM_REGISTRY)}"
            )
        self.latent_sim = latent_sim
        self.sim_kwargs = sim_kwargs or {}

        self._convert_marginals(marginals)

        self.dim_common = dim_common
        self.dim_individual = dim_individual
        self.shared_latent_type = shared_latent_type

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
            mean = np.zeros(self.k)
            g_z = self.rng.multivariate_normal(
                mean=mean, cov=self.column_covariance, size=self.n
            )
            g_e = self.rng.multivariate_normal(
                mean=mean, cov=self.column_covariance, size=self.n
            )
            g_x = self.rho * g_z + np.sqrt(1 - self.rho**2) * g_e

            # 2. Generate Chi-Square variable for scaling
            w = self.rng.chisquare(df=self.df, size=(self.n, 1))

            # 3. Scale to create Multivariate t variables
            scale = np.sqrt(self.df / w)
            t_z = g_z * scale
            t_x = g_x * scale

            # 4. Apply t-distribution CDF to get Uniforms
            u_z = stats.t.cdf(t_z, df=self.df)
            u_x = stats.t.cdf(t_x, df=self.df)

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
            # 1. Assign each sample (row) to a specific mixture component
            # This determines which 'rho' each row will use
            component_indices = self.rng.choice(
                len(self.weights), size=self.n, p=self.weights
            )

            # Initialize the full (n, k) latent Gaussian arrays
            z_full = np.zeros((self.n, self.k))
            x_full = np.zeros((self.n, self.k))

            mean = np.zeros(self.k)

            # 2. Generate data for each mixture component
            for i, rho in enumerate(self.correlations):
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

    def _sample_latent(self):
        """Return X, Z each of shape (n, k)."""

        if self.latent_sim is not None:
            return self._sample_latent_sim()

        if self.dim_common is not None:
            return self._sample_orthogonal_subspace()

        # ── original copula path ──────────────────────────────────────────────
        u_z, u_x = self._generate_copula_uniforms()
        Z = self.marginal_z.ppf(u_z)
        X = self.marginal_x.ppf(u_x)
        while (not np.isfinite(X).all()) or (not np.isfinite(Z).all()):
            X, Z = self._sample_latent()

        if self.center_latent:
            Z = Z - Z.mean(axis=0)
            X = X - X.mean(axis=0)

        return X, Z

    def _sample_orthogonal_subspace(self):
        if not self.dim_individual + self.dim_common == self.k:
            raise ValueError("dim_common + dim_individual must equal k")

        X = np.zeros((self.n, self.k))
        Z = np.zeros((self.n, self.k))

        if self.shared_latent_type == "gaussian":
            U_gauss = np.random.randn(
                self.n, self.dim_common
            )  # Shared latent positions from Gaussian
            V_x_gauss = np.random.randn(
                self.n, self.dim_individual
            )  # X-specific latent positions
            V_z_gauss = np.random.randn(
                self.n, self.dim_individual
            )  # Z-specific latent positions

            U, _ = np.linalg.qr(U_gauss)  # Orthonormalize U
            V_x, _ = np.linalg.qr(V_x_gauss)  # Orthonormalize V_x
            V_z, _ = np.linalg.qr(V_z_gauss)  # Orthonormalize V_z

            X = np.hstack((U, V_x))
            Z = np.hstack((U, V_z))

        elif self.shared_latent_type == "one_hot":
            idxs = np.random.randint(0, self.dim_common, size=self.n)
            C = np.zeros((self.n, self.dim_common))
            C[np.arange(self.n), idxs] = 1.0

            V = np.random.randn(self.n, self.dim_individual)
            W = np.random.randn(self.n, self.dim_individual)

            X = np.concatenate([C, V], axis=1)
            Z = np.concatenate([C, W], axis=1)
        else:
            raise ValueError(f"Unknown shared_latent_type: {self.shared_latent_type}")

        return X, Z

    def _sample_latent_sim(self):
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
            X, Z = self._sample_latent()

        if self.center_latent:
            X = X - X.mean(axis=0)
            Z = Z - Z.mean(axis=0)

        return X, Z

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


class BaseSBM:
    def __init__(
        self,
        n,
        k,
        num_networks=2,
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
        """_summary_

        Parameters
        ----------
        n : int
            Number of nodes
        k : int
            Number of communities
        num_networks : int, optional
            Number of networks, by default 1
        block_probs : _type_, optional
            _description_, by default None
        community_assignment : _type_, optional
            _description_, by default None
        assignment_mode : str, optional
            _description_, by default 'random'
        prob_switch : float, optional
            _description_, by default 0.7
        distance_probs : _type_, optional
            _description_, by default None
        """

        self.n = n
        self.num_networks = num_networks
        if community_assignment is not None:
            k = community_assignment[0].shape[1]

        self.community_assignment = community_assignment
        self.k = k
        self.assignment_mode = assignment_mode
        self.prob_switch = prob_switch
        self.block_probs_type = block_probs_type
        self.block_probs = block_probs
        self.distance_probs = distance_probs
        self.sparsity_bias = sparsity_bias
        self.assortativity = assortativity

    def _sample_community_assignment(self):
        assignment = [np.zeros((self.n, self.k)) for _ in range(self.num_networks)]

        if self.assignment_mode == "random":
            for i in range(self.num_networks):
                idxs = self.rng.integers(low=0, high=self.k, size=self.n)
                assignment[i][np.arange(self.n), idxs] = 1

        elif self.assignment_mode == "correlated":
            first_idxs = self.rng.integers(low=0, high=self.k, size=self.n)
            assignment[0][np.arange(self.n), first_idxs] = 1
            for i in range(1, self.num_networks):
                # with some probability, copy the first assignment, otherwise random
                switch_mask = self.rng.random(self.n) < self.prob_switch
                n_switching_nodes = np.sum(switch_mask)
                if n_switching_nodes > 0:
                    shift = self.rng.integers(1, self.k, size=n_switching_nodes)
                    new_assignment = first_idxs.copy()
                    new_assignment[switch_mask] = (
                        first_idxs[switch_mask] + shift
                    ) % self.k
                    assignment[i][np.arange(self.n), new_assignment] = 1
        else:
            raise ValueError(f"Unknown assignment_mode: {self.assignment_mode}")

        self.community_assignment = assignment
        return assignment

    def _generate_structured_matrix(self, assortativity=None):
        if assortativity is None:
            assortativity = self.assortativity

        # 1. Start with random base probabilities
        mat = self.rng.random((self.k, self.k))

        # Make symmetric (standard for undirected SBMs)
        if self.symmetric:
            mat = (mat + mat.T) / 2.0

        # 2. Create masks to separate diagonal from off-diagonal
        diag_mask = np.eye(self.k, dtype=bool)

        # 3. Apply Assortativity
        # Multiply diagonal by (assortativity * 2) and off-diagonal by ((1 - assortativity) * 2)
        # If assortativity=0.5, both multiply by 1.0 (no change).
        mat[diag_mask] *= assortativity * 2
        mat[~diag_mask] *= (1.0 - assortativity) * 2

        # 4. Apply Sparsity
        mat *= 1 - self.sparsity_bias

        return np.clip(mat, 0.0, 1.0)

    def _sample_block_probs(self):
        probs = []
        if self.block_probs_type == "random":
            probs = [
                self._generate_structured_matrix() for _ in range(self.num_networks)
            ]
        elif self.block_probs_type == "identical":
            # Generate one matrix and use it for all networks
            base_prob = self._generate_structured_matrix()
            probs = [base_prob for _ in range(self.num_networks)]
        elif self.block_probs_type == "correlated":
            base_prob = self._generate_structured_matrix()
            for i in range(1, self.num_networks):
                # Introduce some correlation by adding noise
                noise = self.rng.normal(loc=0.0, scale=0.1, size=base_prob.shape)
                probs.append(np.clip(base_prob + noise, 0.0, 1.0))
        elif self.block_probs_type == "switched":
            for i in range(self.num_networks):
                assortativity = (
                    self.assortativity if i % 2 == 0 else 1 - self.assortativity
                )
                new_prob = self._generate_structured_matrix(assortativity)
                probs.append(new_prob)

        elif self.block_probs_type == "distance":
            raise NotImplementedError

        else:
            raise ValueError(f"Unknown block_probs_type: {self.block_probs_type}")

        self.block_probs = probs

        return probs

    def _sample_sbm_latent(self):
        if self.community_assignment is None:
            community_assignment = self._sample_community_assignment()
            self.community_assignment = community_assignment
        if self.block_probs is None:
            block_probs = self._sample_block_probs()
            self.block_probs = block_probs

        return self.community_assignment, self.block_probs


class GaussianNetwork(CopulaDGP, BaseDPG, BaseSBM):
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
    rho : float
        Correlation parameter for the Gaussian copula (-1 to 1).
    marginals : dict, scipy.stats.rv_continuous
        Marginal distributions for the latent variables. If dict it should have 'z' and 'x' keys.
    edge_var : float, optional
        Variance of the edges (default is 1).
    rng : np.random.Generator, optional
        Random number generator.
    make_sparse: Bool
        Whether to set some of the inner products to zero
    sparsity_bias: float
        Parameter controlling the sparsity of the adjacency matrix.
    """

    def __init__(
        self,
        n,
        k,
        rho=None,
        marginals="gaussian",
        edge_var=1,
        rng=None,
        symmetric=True,
        copula_model="gaussian",
        df=5,
        weights=None,
        correlations=None,
        center_latent=True,
        self_loops=False,
        sparsity_bias=0,
        make_sparse=False,
        column_covariance=None,
        sbm=False,
        latent_X=None,
        latent_Z=None,
        **args,
    ):
        # note here by multiple inheritance CopulaDGP init will be called
        CopulaDGP.__init__(
            self,
            n=n,
            k=k,
            rho=rho,
            marginals=marginals,
            rng=rng,
            copula_model=copula_model,
            df=df,
            weights=weights,
            correlations=correlations,
            center_latent=center_latent,
            column_covariance=column_covariance,
            **args,
        )
        BaseSBM.__init__(self, n=n, k=k, **args)

        self.edge_var = edge_var
        self.symmetric = symmetric
        self.self_loops = self_loops
        self.sparsity_bias = sparsity_bias
        self.make_sparse = make_sparse
        self.sbm = sbm
        self.X = latent_X
        self.Z = latent_Z

    def _make_sparse(self, expected):
        logits = expected - self.sparsity_bias
        probs = 1 / (1 + np.exp(-logits))

        mask = self.rng.uniform(size=expected.shape) < probs
        # mask applied after adj matrix generation, maybe it's worth considering
        # setting the expected to zero directly or taking probs into account for variance
        # computation
        weights = self.rng.normal(loc=expected, scale=self.edge_var)

        return weights * mask

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
            if self.sbm:
                community_assignment, probs_matrices = self._sample_sbm_latent()
                Z_community, Z_probs = community_assignment[0], probs_matrices[0]
                X_community, X_probs = community_assignment[1], probs_matrices[1]

                X = X_community @ X_probs**0.5
                Z = Z_community @ Z_probs**0.5
            else:
                X, Z = self._sample_latent()
                while (not np.isfinite(X).all()) or (not np.isfinite(Z).all()):
                    X, Z = self._sample_latent()

        expected_A = Z @ Z.T
        expected_B = X @ X.T

        if self.make_sparse:
            A = self._make_sparse(expected_A)
            B = self._make_sparse(expected_B)
        else:
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

    def __repr__(self):
        return (
            f"GaussianNetwork(n={self.n}, k={self.k}, rho={self.rho}, "
            f"edge_var={self.edge_var}, "
            f"marginal_z={self.marginal_z}, marginal_x={self.marginal_x})"
        )

    def get_name(self):
        return f"GaussianNetwork_" + str(self.copula_model) + f"_rho{self.rho}"


class BernoulliNetwork(CopulaDGP, BaseDPG, BaseSBM):
    """
    Network Data Generating Process using Bernoulli likelihood.

    Allows for arbitrary marginal distributions for the latent positions Z and X,
    while maintaining a gaussian copula correlation structure. Dot product of latent
    positions models the log-odds of entries of the adjacency matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Dimensionality of the latent space.
    rho : float
        Correlation parameter for the Gaussian copula (-1 to 1).
    marginals : dict, scipy.stats.rv_continuous
        Marginal distributions for the latent variables. If dict it should have 'z' and 'x' keys.
    edge_var : float, optional
        Variance of the edges (default is 1).
    rng : np.random.Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        n,
        k,
        rho=None,
        marginals="gaussian",
        edge_var=1,
        rng=None,
        symmetric=True,
        copula_model="gaussian",
        df=5,
        weights=None,
        correlations=None,
        center_latent=True,
        self_loops=False,
        sparsity_bias=0,
        rdpg=None,
        column_covariance=None,
        sbm=False,
        latent_X=None,
        latent_Z=None,
        **args,
    ):
        CopulaDGP.__init__(
            self,
            n=n,
            k=k,
            rho=rho,
            marginals=marginals,
            rng=rng,
            copula_model=copula_model,
            df=df,
            weights=weights,
            correlations=correlations,
            center_latent=center_latent,
            column_covariance=column_covariance,
            **args,
        )

        BaseSBM.__init__(self, n=n, k=k, **args)

        self.edge_var = edge_var
        self.symmetric = symmetric
        self.self_loops = self_loops
        self.sparsity_bias = sparsity_bias
        self.rdpg = rdpg
        self.sbm = sbm
        self.X = latent_X
        self.Z = latent_Z

    def get_name(self):
        return f"BernoulliNetwork_" + str(self.copula_model) + f"_rho{self.rho}"

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
            if self.sbm:
                community_assignment, probs_matrices = self._sample_sbm_latent()
                Z_community, Z_probs = community_assignment[0], probs_matrices[0]
                X_community, X_probs = community_assignment[1], probs_matrices[1]

                expected_A = Z_community @ Z_probs @ Z_community.T
                expected_B = X_community @ X_probs @ X_community.T

                # placeholder, no real Z and X for SBM
                X, Z = (
                    np.random.normal(size=(self.n, self.k)),
                    np.random.normal(size=(self.n, self.k)),
                )
            else:
                X, Z = self._sample_latent()
                while (not np.isfinite(X).all()) or (not np.isfinite(Z).all()):
                    X, Z = self._sample_latent()

                if self.rdpg is not None:
                    if self.rdpg == "max":
                        # normalise in [-1, 1]
                        X = X / np.max(X, axis=0, keepdims=True)
                        Z = Z / np.max(Z, axis=0, keepdims=True)
                    elif self.rdpg == "spectral":
                        X = X / np.sqrt(np.linalg.norm(X, ord=2))
                        Z = Z / np.sqrt(np.linalg.norm(Z, ord=2))
                    elif self.rdpg == "minmax":
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
                    else:
                        raise Exception(f"Unknown rdpg option: {self.rdpg}")
                    # sparsity applied directly to inner product, maybe worht looking into
                    # a randomly shutting down some edge like weighted network
                    expected_A = Z @ Z.T - self.sparsity_bias
                    expected_B = X @ X.T - self.sparsity_bias
                else:
                    expected_A = expit(Z @ Z.T - self.sparsity_bias)
                    expected_B = expit(X @ X.T - self.sparsity_bias)

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
            print(f"Error generating samples: {e}")
            print(f"Expected probabilities (A): {expected_A}")
            print(f"Expected probabilities (B): {expected_B}")
            raise ValueError

        if self.self_loops is False:
            A[np.diag_indices_from(A)] = 0
            B[np.diag_indices_from(B)] = 0

        out = {"A": A, "B": B, "Z": Z, "X": X}

        return out

    def __repr__(self):
        return (
            f"BernoulliNetwork(n={self.n}, k={self.k}, rho={self.rho}, "
            f"edge_var={self.edge_var}, "
            f"marginal_z={self.marginal_z}, marginal_x={self.marginal_x})"
        )
