import numpy as np
from averagedistorsion.models.model import Model


class ModelGaussianMultimodal(Model):

    def __init__(self, phi=0.2, n_peaks=2):
        self.phi = phi
        self.n_peaks = n_peaks

    def __call__(self, n_voters, n_candidates):
        result = np.zeros((n_voters, n_candidates))
        for i in range(self.n_peaks):
            result += np.random.normal(i/(self.n_peaks - 1), self.phi, size=(n_voters, n_candidates))
        return result / self.n_peaks
