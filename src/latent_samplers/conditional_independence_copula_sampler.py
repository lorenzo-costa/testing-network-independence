"""Stratified copula sampler for conditional-independence simulations."""

from numbers import Integral, Real

import numpy as np

from .copula_sampler import CopulaGenerator


class ConditionalIndependenceCopulaSampler:
    """Sample ``(Z, Y, X)`` from stratum-specific copula models.

    A categorical conditioning variable ``X`` is sampled first. Conditional on
    ``X_i = c``, the corresponding rows of ``Z`` and ``Y`` are sampled from the
    selected copula using ``rho[c]``. Marginals use the same ``{"z", "y"}``
    dictionary format and distribution specifications as :class:`CopulaGenerator`.

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of columns in ``Z``.
    ky : int, default=1
        Number of columns in ``Y``.
    C : int
        Number of categorical strata, labelled ``0, ..., C - 1``.
    rho : float or array-like
        Within-stratum copula correlations. A scalar is repeated for all
        strata; otherwise exactly one value per stratum is required.
    marginals : dict, optional
        Marginal specifications with exactly the required keys ``"z"`` and
        ``"y"``. Defaults to Gaussian marginals for both variables.
    copula_model : {"gaussian", "student_t", "clayton", "gumbel"}
        Copula family used in every stratum.
    copula_params : dict, optional
        Extra copula parameters. ``student_t`` requires a positive ``df``.
    class_probabilities : array-like, optional
        Probabilities for the ``C`` strata. Defaults to uniform probabilities.
    p : array-like, optional
        Alias for ``class_probabilities``.
    center_latent : bool, default=True
        If true, center ``Z`` and ``Y`` separately within every sampled stratum.
    rng : numpy.random.Generator, optional
        Random number generator.

    Notes
    -----
    ``X`` is returned as an integer array with shape ``(n, 1)``. Gaussian and
    Student-t correlations may be negative. Clayton and Gumbel correlations
    must lie in ``[0, 1)``.
    """

    _COPULA_MODELS = {"gaussian", "student_t", "clayton", "gumbel"}

    def __init__(
        self,
        n,
        k,
        C,
        rho=0,
        ky=1,
        marginals=None,
        copula_model="gaussian",
        copula_params=None,
        class_probabilities=None,
        p=None,
        center_latent=True,
        rng=None,
        **copula_kwargs,
    ):
        self.n = self._positive_integer(n, "n")
        self.k = self._positive_integer(k, "k")
        self.ky = self._positive_integer(ky, "ky")
        self.C = self._positive_integer(C, "C")

        if copula_model == "t":
            copula_model = "student_t"
        if copula_model not in self._COPULA_MODELS:
            choices = ", ".join(sorted(self._COPULA_MODELS))
            raise ValueError(f"copula_model must be one of {{{choices}}}.")
        self.copula_model = copula_model

        if marginals is None:
            marginals = {"z": "gaussian", "y": "gaussian"}
        if not isinstance(marginals, dict):
            raise TypeError("marginals must be a dictionary with keys 'z' and 'y'.")
        if set(marginals) != {"z", "y"}:
            raise ValueError("marginals must contain exactly the keys 'z' and 'y'.")
        self.marginals = dict(marginals)

        self.copula_params = dict(copula_params or {})
        if self.copula_model == "student_t":
            df = self.copula_params.get("df")
            if (
                not isinstance(df, Real)
                or isinstance(df, bool)
                or not np.isfinite(df)
                or df <= 0
            ):
                raise ValueError(
                    "copula_params must contain a positive 'df' for student_t."
                )

        if class_probabilities is not None and p is not None:
            raise ValueError("Specify only one of class_probabilities or p.")
        if class_probabilities is None:
            class_probabilities = p

        self.rho = self._normalize_rho(rho)
        self.class_probabilities = self._normalize_probabilities(class_probabilities)
        self.center_latent = bool(center_latent)
        self.rng = rng if rng is not None else np.random.default_rng()
        # Extra fields can arrive from a shared experiment configuration.  The
        # underlying CopulaGenerator accepts and ignores irrelevant keywords.
        self.copula_kwargs = dict(copula_kwargs)
        self._validate_copula_configuration()

        self.X = None
        self.Z = None
        self.Y = None
        self.is_null = bool(
            self.copula_model != "student_t" and np.all(self.rho == 0)
        )

    @staticmethod
    def _positive_integer(value, name):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    def _normalize_rho(self, rho):
        if isinstance(rho, Real) and not isinstance(rho, bool):
            correlations = np.full(self.C, float(rho))
        else:
            correlations = np.asarray(rho, dtype=float)
            if correlations.ndim != 1 or correlations.size != self.C:
                raise ValueError(f"rho must be a scalar or a vector of length C={self.C}.")

        if not np.isfinite(correlations).all():
            raise ValueError("rho must contain only finite values.")
        if self.copula_model in {"clayton", "gumbel"}:
            if np.any(correlations < 0):
                raise ValueError(
                    f"{self.copula_model} does not support negative rho values."
                )
            if np.any(correlations >= 1):
                raise ValueError(
                    f"rho must be less than 1 for the {self.copula_model} copula."
                )
        elif np.any(np.abs(correlations) > 1):
            raise ValueError("rho values must lie in [-1, 1].")
        return correlations

    def _normalize_probabilities(self, probabilities):
        if probabilities is None:
            return np.full(self.C, 1.0 / self.C)

        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 1 or probabilities.size != self.C:
            raise ValueError(
                f"class_probabilities must be a vector of length C={self.C}."
            )
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError(
                "class_probabilities must contain finite, nonnegative values."
            )
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("class_probabilities must sum to 1.")
        return probabilities

    def _validate_copula_configuration(self):
        """Delegate marginal and covariance validation to the base generator."""
        CopulaGenerator(
            n=1,
            k=self.k,
            ky=self.ky,
            rho=self.rho[0],
            marginals=self.marginals,
            copula_model=self.copula_model,
            copula_params=self.copula_params,
            center_latent=False,
            rng=self.rng,
            **self.copula_kwargs,
        )

    def _sample_stratum(self, size, rho):
        # Clayton is singular at theta=0 in the base implementation. Gaussian
        # rho=0 produces the same independent uniforms required at the boundary;
        # using it for Gumbel as well gives an exact, numerically stable limit.
        model = (
            "gaussian"
            if rho == 0 and self.copula_model in {"clayton", "gumbel"}
            else self.copula_model
        )
        generator = CopulaGenerator(
            n=size,
            k=self.k,
            ky=self.ky,
            rho=rho,
            marginals=self.marginals,
            copula_model=model,
            copula_params=self.copula_params,
            center_latent=self.center_latent,
            rng=self.rng,
            **self.copula_kwargs,
        )
        return generator._sample_latent_copula()

    def sample_latent(self):
        """Return ``Z``, ``Y``, and categorical labels ``X``."""
        labels = self.rng.choice(
            self.C,
            size=self.n,
            replace=True,
            p=self.class_probabilities,
        )
        Z = np.empty((self.n, self.k), dtype=float)
        Y = np.empty((self.n, self.ky), dtype=float)

        for stratum, correlation in enumerate(self.rho):
            mask = labels == stratum
            size = int(mask.sum())
            if size == 0:
                continue
            Z[mask], Y[mask] = self._sample_stratum(size, correlation)

        X = labels.reshape(self.n, 1)
        self.Z = Z
        self.Y = Y
        self.X = X
        return Z, Y, X

    def _sample_latent_conditional_copula(self):
        return self.sample_latent()

    def _sample_latent(self):
        return self.sample_latent()

    def sample(self):
        return self.sample_latent()

    def get_name(self):
        return (
            f"conditional_copula_{self.copula_model}_C{self.C}_"
            f"rho{self.rho.tolist()}"
        )
