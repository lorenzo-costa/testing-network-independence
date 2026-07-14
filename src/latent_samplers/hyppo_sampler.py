
import numpy as np
from scipy.special import ndtr

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


class HyppoSimSampler:
    def __init__(
        self,
        n,
        k,
        ky=1,
        sim_name=None,
        sim_kwargs=None,
        center_latent=True,
        make_rdpg=None,
        rng=None,
        **kwargs,
    ):
        self.n = n
        self.k = k
        self.ky = ky
        self.center_latent = center_latent
        self.make_rdpg = make_rdpg

        if (sim_name is not None) and (sim_name not in SIM_REGISTRY):
            raise ValueError(
                f"Unknown sim_name '{sim_name}'. Available: {sorted(SIM_REGISTRY)}"
            )
        self.sim_name = sim_name
        self.sim_kwargs = sim_kwargs or {}
        self.rng = rng or np.random.default_rng()

        if sim_name in ["uncorrelated_bernoulli", "multimodal_independence"]:
            self.is_null = True
        else:
            self.is_null = False

    def _make_rdpg(self, Z, Y):
        """Normalise Z and Y such that inner prods are in [0, 1]"""
        if self.make_rdpg == "max":
            Y = np.abs(Y / np.max(Y, axis=0, keepdims=True))
            Z = np.abs(Z / np.max(Z, axis=0, keepdims=True))
        elif self.make_rdpg == "spectral":
            Y = Y / np.sqrt(np.linalg.norm(Y, ord=2))
            Z = Z / np.sqrt(np.linalg.norm(Z, ord=2))
        elif self.make_rdpg == "minmax":
            Y = (Y - np.min(Y, axis=0, keepdims=True)) / (
                np.max(Y, axis=0, keepdims=True)
                - np.min(Y, axis=0, keepdims=True)
                + 1e-15
            )
            Y = Y / np.sqrt(Y.shape[1])

            Z = (Z - np.min(Z, axis=0, keepdims=True)) / (
                np.max(Z, axis=0, keepdims=True)
                - np.min(Z, axis=0, keepdims=True)
                + 1e-15
            )
            Z = Z / np.sqrt(Z.shape[1])

        elif self.make_rdpg == "hypersphere":
            # Map latent values to positive orthant directions
            V_X = ndtr(Y)
            V_Z = ndtr(Z)

            norm_X = np.linalg.norm(V_X, axis=1, keepdims=True)
            norm_Z = np.linalg.norm(V_Z, axis=1, keepdims=True)

            dir_X = np.where(norm_X > 0, V_X / norm_X, 0)
            dir_Z = np.where(norm_Z > 0, V_Z / norm_Z, 0)

            # Use one copula-derived coordinate for the radius
            u_y = ndtr(Y[:, 0:1])
            u_z = ndtr(Z[:, 0:1])

            r_X = u_y ** (1.0 / self.k)
            r_Z = u_z ** (1.0 / self.k)

            Y = r_X * dir_X
            Z = r_Z * dir_Z
        else:
            raise Exception(f"Unknown rdpg option: {self.make_rdpg}")

        return Z, Y

    def _sample_latent_hyppo(self):
        """
        Use one of the simulation functions to produce (Y, Z).

        The sim function is called as  sim(n, k, **sim_kwargs).
        Its two return values are treated as (Y, Z):
          - first return  → Y  (the 'input' latent positions)
          - second return → Z  (the 'output' latent positions)

        Shape alignment
        ---------------
        Some sims return y with shape (n, 1) instead of (n, k).
        We tile those to (n, k) so downstream code always sees (n, k).
        If the sim returns something wider than k we trim to the first k columns.
        """
        sim_fn = SIM_REGISTRY[self.sim_name]

        raw_y, raw_z = sim_fn(n=self.n, p=self.k, **self.sim_kwargs)
        Y = self._align_shape(np.asarray(raw_y, dtype=float), self.ky)
        Z = self._align_shape(np.asarray(raw_z, dtype=float), self.k)

        while (not np.isfinite(Y).all()) or (not np.isfinite(Z).all()):
            raw_y, raw_z = sim_fn(n=self.n, p=self.k, **self.sim_kwargs)
            Y = self._align_shape(np.asarray(raw_y, dtype=float), self.ky)
            Z = self._align_shape(np.asarray(raw_z, dtype=float), self.k)

        if self.center_latent:
            Y = Y - Y.mean(axis=0)
            Z = Z - Z.mean(axis=0)

        if self.make_rdpg is not None:
            Z, Y = self._make_rdpg(Z, Y)

        return Z, Y

    def _align_shape(self, arr, dimension):
        """Ensure ``arr`` has the requested number of columns."""
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        _, cols = arr.shape

        if cols == dimension:
            return arr
        if cols < dimension:
            # tile: repeat columns until we reach k, then trim
            repeats = -(-dimension // cols)  # ceiling division
            arr = np.tile(arr, (1, repeats))
        return arr[:, :dimension]

    def get_name(self):
        return f"HyppoSim_{self.sim_name}"
