import numpy as np
from averagedistorsion.models.model_euclidean import ModelEuclidean


class ModelEuclideanGaussian(ModelEuclidean):

    def __init__(self, loc=0.5, phi=0.2, dim=2, norm=False):
        super().__init__(dim=dim, norm=norm)
        self.phi = phi
        self.loc = loc

    def generate_points(self, n_points):
        return np.random.normal(self.loc, self.phi, size=(n_points, self.dim))
