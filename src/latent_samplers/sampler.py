
import warnings
from numbers import Integral

from .copula_sampler import CopulaGenerator
from .conditional_independence_copula_sampler import (
    ConditionalIndependenceCopulaSampler,
)
from .post_nonlinear_noise_sampler import PostNonLinearNoiseSampler

try:
    from .hyppo_sampler import HyppoSimSampler
except ImportError:
    HyppoSimSampler = None
    warnings.warn(
        "Hyppo package not found. HyppoSimSampler will not be available. Please install hyppo to use this feature.",
        ImportWarning,
        stacklevel=2,
    )

from .orthogonal_subspace import OrthogonalSubspaceSampler
from .rdpg_sampler import RDPGGenerator
from .functional_sampler import FunctionalGenerator
from .sbm_cov_sampler import SBMCovariateGenerator
import numpy as np


import warnings


class LatentSampler:
    """Base class for sampling latent variables

    Arguments
    ----------
    n : int
        Number of samples (nodes).
    k : int
        Dimensionality of the latent space.
    conditional_copula : str or True, optional
        Select the conditional copula sampler. A string names the copula model;
        with True, ``copula_model`` is used and defaults to ``"gaussian"``.
    copula_model : str
        Type of copula to use for generating dependence structure. If not None, function sample from Copula
    hyppo_sim : str
        Name of a simulation function from hyppo package to use for generating latent variables instead of the copula path.
        If not None function samples from specified sim function. Takes precedence over copula_model if both are provided.
    dim_common : int
        If not None Y and Z are sampled with some shared + individual structure.
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
        ky=1,
        conditional_copula=None,
        post_nonlinear_noise=None,
        copula_model=None,
        latent_sim=None,
        dim_common=None,
        block_probs_type=None,
        rdpg_distr=None,
        functional_form=None,
        sbm_covariate_sampling=None,
        rng=None,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng
        self.n = n
        self.k = k
        if not isinstance(ky, Integral) or isinstance(ky, bool) or ky < 1:
            raise ValueError("ky must be a positive integer.")
        self.ky = int(ky)

        sampler_options = [
            ("conditional_copula", conditional_copula),
            ("post_nonlinear_noise", post_nonlinear_noise),
            ("sbm_covariate_sampling", sbm_covariate_sampling),
            ("functional_form", functional_form),
            ("latent_sim", latent_sim),
            ("dim_common", dim_common),
            ("block_probs_type", block_probs_type),
            ("rdpg_distr", rdpg_distr),
            (
                "copula_model",
                copula_model
                if conditional_copula is None and post_nonlinear_noise is None
                else None,
            ),
        ]


        selected_name, selected_value = next(
            ((name, value) for name, value in sampler_options if value is not None),
            ("copula_model", None),  # default construction
        )

        ignored = [
            name
            for name, value in sampler_options
            if value is not None and name != selected_name
        ]

        if ignored:
            warnings.warn(
                f"{selected_name} was specified; ignoring: {', '.join(ignored)}.",
                UserWarning,
                stacklevel=2,
            )

        factories = {
            "conditional_copula": lambda value: ConditionalIndependenceCopulaSampler(
                n=n,
                k=k,
                ky=ky,
                copula_model=self._conditional_copula_model(value, copula_model),
                rng=rng,
                **kwargs,
            ),
            "post_nonlinear_noise": lambda value: PostNonLinearNoiseSampler(
                n=n,
                k=k,
                ky=ky,
                rng=rng,
                **self._post_nonlinear_kwargs(value, kwargs),
            ),
            "sbm_covariate_sampling": lambda value: SBMCovariateGenerator(
                n=n, k=k, ky=ky, sampling=value, rng=rng, **kwargs
            ),
            "functional_form": lambda value: FunctionalGenerator(
                n=n, k=k, ky=ky, functional_form=value, rng=rng, **kwargs
            ),
            "latent_sim": lambda value: HyppoSimSampler(
                n=n, k=k, ky=ky, sim_name=value, rng=rng, **kwargs
            ) if HyppoSimSampler is not None else self._hyppo_unavailable(),
            "dim_common": lambda value: OrthogonalSubspaceSampler(
                n=n, k=k, ky=ky, dim_common=value, rng=rng, **kwargs
            ),
            "rdpg_distr": lambda value: RDPGGenerator(
                n=n, k=k, ky=ky, rdpg_distr=value, rng=rng, **kwargs
            ),
            "copula_model": lambda value: CopulaGenerator(
                n=n, k=k, ky=ky, copula_model=value, rng=rng, **kwargs
            ),
        }

        latent_sampler = factories[selected_name](selected_value)

        self.latent_sampler = latent_sampler
        self.sampler_name = latent_sampler.get_name()

        self.copula_model = copula_model
        self.conditional_copula = conditional_copula
        self.post_nonlinear_noise = post_nonlinear_noise
        self.latent_sim = latent_sim
        self.dim_common = dim_common
        self.block_probs_type = block_probs_type
        self.rdpg_distr = rdpg_distr
        self.sbm_covariate_sampling = sbm_covariate_sampling

        self.functional_form = functional_form
        self.X = None

    @staticmethod
    def _hyppo_unavailable():
        raise ImportError("The optional 'hyppo' package is required for latent_sim.")

    @staticmethod
    def _conditional_copula_model(activation, copula_model):
        if isinstance(activation, str):
            return activation
        if activation is True:
            return copula_model or "gaussian"
        raise ValueError(
            "conditional_copula must be a copula-model string or True."
        )

    @staticmethod
    def _post_nonlinear_kwargs(activation, kwargs):
        if activation is not True:
            raise ValueError("post_nonlinear_noise must be True when selected.")
        sampler_kwargs = dict(kwargs)
        sampler_kwargs.pop("rdpg", None)
        return sampler_kwargs

    def _sample_latent(self):
        """Return ``(Z, Y, X)``; ``X`` is ``None`` when not generated.
        Hierarchy of generation is:
        - if conditional_copula is specified, generate (Z, Y, X) by stratum and
          retain X on this wrapper.
        - if functional_form is specified, use that to generate (Z, Y) directly.
        - else if hyppo_sim is specified, use that to generate (Z, Y) directly.
        - else if dim_common is specified, generate Z and Y with shared + individual structure.
        - else if block_probs_type is specified, generate Z and Y with SBM structure.
        - else use the copula-based generation with the specified marginals and dependence structure.
        """
        Z, Y, X = None, None, None

        if self.conditional_copula is not None:
            Z, Y, X = self.latent_sampler._sample_latent_conditional_copula()
        elif self.post_nonlinear_noise is not None:
            Z, Y, X = self.latent_sampler._sample_latent_post_nonlinear_noise()
        elif self.sbm_covariate_sampling is not None:
            Y, community_assignment, block_probs = self.latent_sampler._sample_latent_sbm_covariate()
            Z = community_assignment @ block_probs**0.5
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
        elif self.functional_form is not None:
            Z, Y = self.latent_sampler._sample_latent_functional()
        elif self.latent_sim is not None:
            Z, Y = self.latent_sampler._sample_latent_hyppo()
        elif self.dim_common is not None:
            Z, Y = self.latent_sampler._sample_latent_orthogonal()
        elif self.block_probs_type is not None:
            (
                community_assignment_z,
                community_assignment_x,
                probs_matrix_z,
                probs_matrix_x,
            ) = self.latent_sampler._sample_sbm_latent()
            Y = community_assignment_x @ probs_matrix_x**0.5
            Z = community_assignment_z @ probs_matrix_z**0.5
        elif self.rdpg_distr is not None:
            Z, Y = self.latent_sampler._sample_latent_rdpg()
        elif self.copula_model is not None:
            Z, Y = self.latent_sampler._sample_latent_copula()
        else:
            raise ValueError(
                "No valid latent generation method specified. Please provide one of: conditional_copula, post_nonlinear_noise, latent_sim, dim_common, block_probs_type, or copula_model."
            )

        Z = np.asarray(Z)
        Y = np.asarray(Y)
        if Z.ndim != 2 or Z.shape != (self.n, self.k):
            raise ValueError(f"Z must have shape ({self.n}, {self.k}); got {Z.shape}.")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.ndim != 2 or Y.shape != (self.n, self.ky):
            raise ValueError(f"Y must have shape ({self.n}, {self.ky}); got {Y.shape}.")
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2 or X.shape != (self.n, 1):
                raise ValueError(
                    f"X must have shape ({self.n}, 1); got {X.shape}."
                )
        self.X = X
        self.is_null = getattr(self.latent_sampler, "is_null", None)
        return Z, Y, X

    def sample_latent(self):
        return self._sample_latent()
