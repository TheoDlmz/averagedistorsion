import numpy as np
from averagedistorsion.models.model import Model


class ModelEuclidean(Model):
    """
    Euclidean models are models in which utilities are represented by the opposite of the distance between a voter
    and a candidate. The goal is then to minimize the sum of distance to the winning candidate.

     Parameters
    ----------
    dim: Int
        The number of dimensions of the Euclidean space. Default is 2
    norm: Bool
        If true, utilities are normalized

    """

    positive = False

    def __init__(self, dim=2, norm=False, fixed=False, positions_irrelevant=None):
        self.dim = dim
        self.norm = norm
        self.fixed = fixed
        self.positions_irrelevant = positions_irrelevant

    def generate_points(self, n_points):
        """

        Parameters
        ----------
        n_points: Int
            The number of points to generate on the Euclidean space

        Returns
        -------
        np.array
            The positions of the points on the plane

        """
        raise NotImplementedError

    def __call__(self, n_voters, n_candidates):
        p_voters = self.generate_points(n_voters)
        if self.fixed:
            p_candidates_relevant = np.zeros((2, self.dim))
            p_candidates_relevant[1][:] = 1
        else:
            p_candidates_relevant = self.generate_points(2)

        if self.positions_irrelevant is None:
            p_candidates_irrelevant = self.generate_points(n_candidates)
        else:
            p_candidates_irrelevant = self.positions_irrelevant(n_candidates,
                                                                p_candidates_relevant)

        if len(p_candidates_irrelevant) > 0:
            # print(p_candidates_irrelevant, p_candidates_relevant)
            # print(p_candidates_irrelevant.shape, p_candidates_relevant.shape)

            p_candidates = np.concatenate([p_candidates_relevant, p_candidates_irrelevant], axis=0)
        else:
            p_candidates = p_candidates_relevant
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
