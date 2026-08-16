
import numpy as np

class RDPGGenerator:
    """Base class for sampling continuous latent positions for an RDPG.

    Parameters
    ----------
    n : int
        Number of samples (nodes).
    k : int
        Dimensionality of the latent space.
    """

    def __init__(self):
        pass
        
    def _sample_latent_rdpg(self):
        pass


    def get_name(self):
        return f"RDPG_{self.rdpg_distr}"
