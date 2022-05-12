import numpy as np
from averagedistorsion.models.model import Model


class ModelEuclidean(Model):

    def __init__(self, dim=2, norm=False):
        self.dim = dim
        self.norm = norm

    def generate_points(self, n_points):
        raise NotImplementedError

    def __call__(self, n_voters, n_candidates):
        p_voters = self.generate_points(n_voters)
        p_candidates = self.generate_points(n_candidates)
        result = np.zeros((n_voters, n_candidates))
        for i in range(n_voters):
            for j in range(n_candidates):
                dist = np.sqrt(sum((p_voters[i][k] - p_candidates[j][k])**2 for k in range(self.dim)))
                # round_dist = np.ceil(dist*100)/100 # To avoid infinite utilities
                # result[i, j] = 1 / round_dist
                result[i, j] = -dist
        if self.norm:
            result = (result.T / result.sum(axis=1)).T
        return result
