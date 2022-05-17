import numpy as np
from averagedistorsion.models.model import Model


class ModelGaussian(Model):
    """
    Model in which utilities are drawn from a Gaussian

    Parameters
    ----------
    phi: float
        The std of the Gaussian
    """

    def __init__(self, phi=0):
        self.phi = phi

    def __call__(self, n_voters, n_candidates):
        voter_pref = np.random.rand(n_candidates)
        matrix_id = np.stack([voter_pref for _ in range(n_voters)])
        return matrix_id + np.random.normal(0, self.phi, size=(n_voters, n_candidates))
