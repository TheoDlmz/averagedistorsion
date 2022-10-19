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

    def __init__(self, phi=0.1, center=None):
        self.phi = phi
        if center is None:
            self.center = None
        else:
            self.center = np.array(center)

    def __call__(self, n_voters, n_candidates):
        if self.center is None:
            voter_pref = np.random.rand(n_candidates)
        else:
            voter_pref = self.center*np.ones(n_candidates)

        matrix_id = np.stack([voter_pref for _ in range(n_voters)])
        output = matrix_id + np.random.normal(0, self.phi, size=(n_voters, n_candidates))
        return np.maximum(0, output)
