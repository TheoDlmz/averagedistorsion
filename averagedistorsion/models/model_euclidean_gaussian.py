import numpy as np
from averagedistorsion.models.model_euclidean import ModelEuclidean


class ModelEuclideanGaussian(ModelEuclidean):
    """
    An Euclidean model in which points are generated using a gaussian

    Parameters
    ----------
    loc: float
        The center of the gaussian
    phi: float
        The std of the gaussian
    dim: int
        The number of dimensions of the Euclidean space
    norm: bool
        If True, the utilities are normalized

    """

    def __init__(self, loc=0.5, phi=0.2, dim=2, norm=False):
        super().__init__(dim=dim, norm=norm)
        self.phi = phi
        self.loc = loc

    def generate_points(self, n_points):
        return np.random.normal(self.loc, self.phi, size=(n_points, self.dim))
