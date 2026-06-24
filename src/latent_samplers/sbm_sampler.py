
import numpy as np

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
        assignment_mode=None,
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

        self.is_null = True

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
                new_assignment[switch_mask] = (idxs_z[switch_mask] + shift) % self.kz

            assignment_x[np.arange(self.n), new_assignment] = 1

            self.is_null = False
        else:
            raise ValueError(f"Unknown assignment_mode: {self.assignment_mode}")

        self.assignment_z = assignment_z
        self.assignment_x = assignment_x

        return assignment_z, assignment_x

    def _generate_probability_matrix(self, k, assortativity=None):
        if assortativity is None:
            assortativity = self.assortativity

        mat = self.rng.random((k, k))

        if self.symmetric:
            mat = (mat + mat.T) / 2.0

        diag_mask = np.eye(k, dtype=bool)

        # If assortativity=0.5, both multiply by 1.0 (no change).
        mat[diag_mask] *= assortativity * 2
        mat[~diag_mask] *= (1.0 - assortativity) * 2

        mat *= 1 - self.sparsity_bias

        return np.clip(mat, 0.0, 1.0)

    def _sample_block_probs(self):
        if self.block_probs_type == "random":
            probs_x = self._generate_probability_matrix(self.kx)
            probs_z = self._generate_probability_matrix(self.kz)
        elif self.block_probs_type == "identical":
            # Generate one matrix and use it for all networks
            if self.kx != self.kz:
                raise ValueError(
                    "For 'identical' block_probs_type, kx and kz must be the same."
                )
            probs_z = self._generate_probability_matrix(self.kx)
            probs_x = probs_z.copy()

            self.is_null = False

        elif self.block_probs_type == "correlated":
            if self.kx != self.kz:
                raise ValueError(
                    "For 'correlated' block_probs_type, kx and kz must be the same."
                )
            probs_z = self._generate_probability_matrix(self.kz)
            for i in range(1, self.num_networks):
                # Introduce some correlation by adding noise
                noise = self.rng.normal(loc=0.0, scale=0.1, size=probs_z.shape)
                probs_x.append(np.clip(probs_z + noise, 0.0, 1.0))

            self.is_null = False

        elif self.block_probs_type == "switched":
            if self.kx != self.kz:
                raise ValueError(
                    "For 'switched' block_probs_type, kx and kz must be the same."
                )
            probs_z = self._generate_probability_matrix(self.kz, self.assortativity)
            probs_x = self._generate_probability_matrix(self.kx, 1 - self.assortativity)

            self.is_null = False

        elif self.block_probs_type == "distance":
            raise NotImplementedError

        else:
            raise ValueError(f"Unknown block_probs_type: {self.block_probs_type}")

        self.block_probs_x = probs_x
        self.block_probs_z = probs_z

        return probs_z, probs_x

    def _sample_sbm_latent(self):
        if self.community_assignment is None:
            community_assignment_z, community_assignment_x = (
                self._sample_community_assignment()
            )
            self.community_assignment_z = community_assignment_z
            self.community_assignment_x = community_assignment_x
        if self.block_probs is None:
            block_probs_z, block_probs_x = self._sample_block_probs()
            self.block_probs_z = block_probs_z
            self.block_probs_x = block_probs_x

        return (
            self.community_assignment_z,
            self.community_assignment_x,
            self.block_probs_z,
            self.block_probs_x,
        )
