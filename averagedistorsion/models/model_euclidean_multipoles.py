import numpy as np
from averagedistorsion.models.model_euclidean import ModelEuclidean


class ModelEuclideanMultiPoles(ModelEuclidean):
    """
    An Euclidean model in which position are drawn from several gaussian. Their position are
    drawn randomly on the plane.

    Parameters
    ----------
    poles_num: int
        The number of gaussian to use
    phi: float
        The std of the gaussian
    dim: int
        The number of dimensions of the Euclidean space
    norm: bool
        If True, the utilities are normalized

    """

    def __init__(self, poles_num=3, phi=0.2, dim=2, norm=False):
        super().__init__(dim=dim, norm=norm)
        self.poles_num = poles_num
        self.dim = dim
        self.phi = phi

    def generate_points(self, n_points):
        poles_points = np.random.rand(self.poles_num, self.dim)
        poles_weights = np.random.rand(self.poles_num)
        poles_weights_sum = poles_weights.sum()
        poles_sizes = [int(x * n_points / poles_weights_sum) for x in poles_weights]
        while sum(poles_sizes) < n_points:
            i  = np.random.randint(self.poles_num)
            poles_sizes[i] += 1
        points = np.zeros((n_points, self.dim))
        start = 0
        for i in range(self.poles_num):
            for j in range(poles_sizes[i]):
                for k in range(self.dim):
                    points[start + j][k] = np.random.normal(loc = poles_points[i][k], scale = self.phi)
            start += poles_sizes[i]
        return points
