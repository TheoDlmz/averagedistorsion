import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class Rule(DeleteCacheMixin):

    def __call__(self, matrix):
        self.delete_cache()
        self.matrix_ = matrix
        return self

    def majorityMatrix(self):
        n, m = self.matrix_.shape
        newMatrix = np.zeros((m, m))
        for row in self.matrix_:
            r = np.argsort(row)
            for i in range(m):
                for j in range(i + 1, m):
                    newMatrix[r[j], r[i]] += 1
                    newMatrix[r[i], r[j]] -= 1
        return newMatrix

    @cached_property
    def winner_(self):
        raise NotImplementedError

    @cached_property
    def utilities_(self):
        return self.matrix_.sum(0)

    @cached_property
    def distortion_(self):
        util = self.utilities_
        return max(1, np.max(util) / util[self.winner_])

    @cached_property
    def cost_(self):
        util = -self.utilities_
        return max(1, util[self.winner_]/np.min(util))

