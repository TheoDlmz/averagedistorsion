import numpy as np
from averagedistorsion.models.model import Model


class ModelUniform(Model):

    def __call__(self, n_voters, n_candidates):
        return np.random.rand(n_voters, n_candidates)
