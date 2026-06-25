
from .copula_sampler import CopulaGenerator
from .hyppo_sampler import HyppoSimSampler
from .orthogonal_subspace import OrthogonalSubspaceSampler
from .rdpg_sampler import RDPGGenerator
from .sbm_sampler import SBMGenerator
import numpy as np

class LatentSampler:
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
        kx=None,
        copula_model=None,
        latent_sim=None,
        dim_common=None,
        block_probs_type=None,
        rdpg_distr=None,
        rng=None,
        force_x_single_dimension=False,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng
        self.n = n
        self.k = k
        

        if latent_sim is not None:
            if copula_model is not None:
                raise Warning(
                    "Both latent_sim and copula_model specified. copula_model will be ignored."
                )
            if dim_common is not None:
                raise Warning(
                    "Both latent_sim and dim_common specified. dim_common will be ignored."
                )
            if block_probs_type is not None:
                raise Warning(
                    "Both latent_sim and block_probs_type specified. block_probs_type will be ignored."
                )
            if rdpg_distr is not None:
                raise Warning(
                    "Both latent_sim and rdpg_distr specified. rdpg_distr will be ignored."
                )

            latent_sampler = HyppoSimSampler(
                n=n, k=k, sim_name=latent_sim, rng=rng, **kwargs
            )

        elif dim_common is not None:
            if copula_model is not None:
                raise Warning(
                    "Both dim_common and copula_model specified. copula_model will be ignored."
                )
            if block_probs_type is not None:
                raise Warning(
                    "Both dim_common and block_probs_type specified. block_probs_type will be ignored."
                )
            if rdpg_distr is not None:
                raise Warning(
                    "Both dim_common and rdpg_distr specified. rdpg_distr will be ignored."
                )

            latent_sampler = OrthogonalSubspaceSampler(
                n=n, k=k, dim_common=dim_common, rng=rng, **kwargs
            )

        elif block_probs_type is not None:
            if copula_model is not None:
                raise Warning(
                    "Both block_probs_type and copula_model specified. copula_model will be ignored."
                )
            if rdpg_distr is not None:
                raise Warning(
                    "Both block_probs_type and rdpg_distr specified. rdpg_distr will be ignored."
                )

            latent_sampler = SBMGenerator(
                n=n, k=k, kx=kx, rng=rng, block_probs_type=block_probs_type, **kwargs
            )

        elif rdpg_distr is not None:
            if copula_model is not None:
                raise Warning(
                    "Both rdpg_distr and copula_model specified. copula_model will be ignored."
                )

            latent_sampler = RDPGGenerator(
                n=n, k=k, kx=kx, rng=rng, rdpg_distr=rdpg_distr, **kwargs
            )

        else:
            latent_sampler = CopulaGenerator(
                n=n, k=k, kx=kx, rng=rng, copula_model=copula_model, **kwargs
            )

        self.latent_sampler = latent_sampler
        self.sampler_name = latent_sampler.get_name()

        self.copula_model = copula_model
        self.latent_sim = latent_sim
        self.dim_common = dim_common
        self.block_probs_type = block_probs_type
        self.rdpg_distr = rdpg_distr

        self.force_x_single_dimension = force_x_single_dimension

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
            Z, X = self.latent_sampler._sample_latent_hyppo()
        elif self.dim_common is not None:
            Z, X = self.latent_sampler._sample_latent_orthogonal()
        elif self.block_probs_type is not None:
            (
                community_assignment_z,
                community_assignment_x,
                probs_matrix_z,
                probs_matrix_x,
            ) = self.latent_sampler._sample_sbm_latent()
            X = community_assignment_x @ probs_matrix_x**0.5
            Z = community_assignment_z @ probs_matrix_z**0.5
        elif self.rdpg_distr is not None:
            Z, X = self.latent_sampler._sample_latent_rdpg()
        elif self.copula_model is not None:
            Z, X = self.latent_sampler._sample_latent_copula()
        else:
            raise ValueError(
                "No valid latent generation method specified. Please provide one of: latent_sim, dim_common, block_probs_type, or copula_model."
            )

        if self.force_x_single_dimension:
            X = X[:, 0:1]  # Keep only the first dimension of X

        return Z, X