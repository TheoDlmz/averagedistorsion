import numpy as np
from averagedistorsion.models.model_metric.model_euclidean import ModelEuclidean


class ModelEuclideanUniform(ModelEuclidean):
    """
    An Euclidean model in which position are drawn uniformly.
    """
    def generate_points(self, n_points):
        return np.random.rand(n_points, self.dim)
