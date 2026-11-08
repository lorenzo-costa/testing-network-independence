
import warnings
from numbers import Integral

from .copula_sampler import CopulaGenerator
from .conditional_independence_copula_sampler import (
    ConditionalIndependenceCopulaSampler,
)
from .post_nonlinear_noise_sampler import PostNonLinearNoiseSampler
from .rdpg_sampler import RDPGGenerator
from .functional_sampler import FunctionalGenerator
import numpy as np


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
    conditional_gaussian : True, optional
        Select the categorical joint Gaussian sampler. Requires ``means``,
        ``covariances``, and ``class_probabilities`` in keyword arguments.
    copula_model : str
        Type of copula to use for generating dependence structure. If not None, function sample from Copula
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
        rdpg_distr=None,
        functional_form=None,
        rng=None,
        conditional_gaussian=None,
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
            ("conditional_gaussian", conditional_gaussian),
            ("conditional_copula", conditional_copula),
            ("post_nonlinear_noise", post_nonlinear_noise),
            ("functional_form", functional_form),
            ("rdpg_distr", rdpg_distr),
            (
                "copula_model",
                copula_model
                if conditional_gaussian is None and conditional_copula is None and post_nonlinear_noise is None
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
            "conditional_gaussian": lambda value: ConditionalGaussianSampler(
                n=n,
                k=k,
                ky=ky,
                rng=rng,
                **self._conditional_gaussian_kwargs(value, kwargs),
            ),
            "post_nonlinear_noise": lambda value: PostNonLinearNoiseSampler(
                n=n,
                k=k,
                ky=ky,
                rng=rng,
                **self._post_nonlinear_kwargs(value, kwargs),
            ),
            "functional_form": lambda value: FunctionalGenerator(
                n=n, k=k, ky=ky, functional_form=value, rng=rng, **kwargs
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
        self.conditional_gaussian = conditional_gaussian
        self.post_nonlinear_noise = post_nonlinear_noise
        self.rdpg_distr = rdpg_distr

        self.functional_form = functional_form
        self.X = None

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
        allowed = {
            "C",
            "rho",
            "error_covariance",
            "stratum_covariance",
            "column_covariance_z",
            "column_covariance_y",
            "column_covariance",
            "cross_correlation_template",
            "class_probabilities",
            "p",
            "center_latent",
            "function_type",
        }
        return {key: value for key, value in kwargs.items() if key in allowed}

    @staticmethod
    def _conditional_gaussian_kwargs(activation, kwargs):
        if activation is not True:
            raise ValueError("conditional_gaussian must be True when selected.")
        return {
            key: kwargs[key]
            for key in ("means", "covariances", "class_probabilities")
            if key in kwargs
        }

    def _sample_latent(self):
        """Return ``(Z, Y, X)``; ``X`` is ``None`` when not generated.
        Hierarchy of generation is:
        - if conditional_gaussian is specified, generate (Z, Y, X) from the
          explicitly configured category Gaussians.
        - if conditional_copula is specified, generate (Z, Y, X) by stratum and
          retain X on this wrapper.
        - if functional_form is specified, use that to generate (Z, Y) directly.
        - else use the copula-based generation with the specified marginals and dependence structure.
        """
        Z, Y, X = None, None, None

        if self.conditional_gaussian is not None:
            Z, Y, X = self.latent_sampler.sample_latent()
        elif self.conditional_copula is not None:
            Z, Y, X = self.latent_sampler._sample_latent_conditional_copula()
        elif self.post_nonlinear_noise is not None:
            Z, Y, X = self.latent_sampler._sample_latent_post_nonlinear_noise()
        elif self.functional_form is not None:
            Z, Y = self.latent_sampler._sample_latent_functional()
        elif self.rdpg_distr is not None:
            Z, Y = self.latent_sampler._sample_latent_rdpg()
        elif self.copula_model is not None:
            Z, Y = self.latent_sampler._sample_latent_copula()
        else:
            raise ValueError(
                "No valid latent generation method specified. Please provide one of: conditional_gaussian, conditional_copula, post_nonlinear_noise, functional_form, rdpg_distr, or copula_model."
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
