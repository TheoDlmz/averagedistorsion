import numpy as np


class positionsFunction:

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, n_candidates, pos_relevant):
        raise NotImplementedError


class positionsFunctionEquidistant(positionsFunction):

    def __call__(self, n_candidates, pos_relevant):
        [self.pos_a, self.pos_b] = pos_relevant
        positions = np.zeros((self.dim, n_candidates-2))
        if n_candidates > 2:
            positions_inter = (self.pos_a + np.array([x*(self.pos_b-self.pos_a) for x in np.linspace(0, 1, n_candidates)]))
            positions = positions_inter[1:n_candidates-1]
            return positions
        else:
            return []




## The idea is to generate positions of the irrelevant alternatives from the position of the relevant alternatives
# with different models. However, I do not know if we fix the position of the relevant alternatives.
