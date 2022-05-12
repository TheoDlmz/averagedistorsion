import numpy as np
from averagedistorsion.models.model_euclidean import ModelEuclidean


class ModelEuclideanUniform(ModelEuclidean):

    def generate_points(self, n_points):
        return np.random.rand(n_points, self.dim)
