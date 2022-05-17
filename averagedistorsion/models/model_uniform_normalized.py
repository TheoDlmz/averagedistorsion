import numpy as np
from averagedistorsion.models.model import Model


class ModelUniformNormalized(Model):
    """
    Model in which utilities are drawn uniformly but are normalized for each voter
    """

    def __call__(self, n_voters, n_candidates):
        pos = np.random.rand(n_voters, n_candidates)
        return (pos.T / pos.sum(axis=1)).T
