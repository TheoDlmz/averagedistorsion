import numpy as np
from averagedistorsion.models.model import Model


class ModelIdentical(Model):
    """
    Model in which candidates have a true utility and voter have a noisy approximation of it

    Parameters
    ----------
    phi: float
        The % of noise is phi
    """
    def __init__(self, phi=0):
        self.phi = phi

    def __call__(self, n_voters, n_candidates):
        voter_pref = np.random.rand(n_candidates)
        matrix_id = np.stack([voter_pref for _ in range(n_voters)])
        return (1-self.phi)*matrix_id + self.phi*np.random.rand(n_voters, n_candidates)
