import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class Rule(DeleteCacheMixin):
    """
    The family of classes for aggregating rules
    """

    def __call__(self, matrix):
        self.delete_cache()
        self.matrix_ = matrix
        return self

    def majorityMatrix(self):
        """
        Compute the majority matrix, i.e. the matrix that contains for every pairs of candidates (i,j) the relative
        number of voters that prefers candidate i to candidate j

        Returns
        -------
        np.array
            The majority matrix

        """
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
        """
        Returns
        -------
        int
            The id of the winning candidate

        """
        raise NotImplementedError

    @cached_property
    def utilities_(self):
        """
        Returns
        -------
        np.array
            The sum of utilities for every candidate

        """
        return self.matrix_.sum(0)

    @cached_property
    def distortion_(self):
        """
        Return the distortion of the winning candidate, which is the maximum utility divided by the utility of
        the winning candidate, and is 1 if the winning candidate is the best one.
        Returns
        -------
        float
            The distortion of the winning candidate

        """
        util = self.utilities_
        return max(1, np.max(util) / util[self.winner_])

    @cached_property
    def cost_(self):
        """
        Return the distortion cost of the winning candidate, which is the cost of the winning candidate divided by the
        minimum cost (in case of negative utilities, i.e. costs), and is 1 if the winning candidate is the best one.
        Returns
        -------
        float
            The distortion cost of the winning candidate

        """
        util = -self.utilities_
        return max(1, util[self.winner_]/np.min(util))

