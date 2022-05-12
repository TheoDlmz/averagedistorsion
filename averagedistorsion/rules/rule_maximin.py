import numpy as np
from averagedistorsion.rules.rule import Rule
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleMaximin(Rule):

    name = "Maximin"

    @cached_property
    def winner_(self):
        n, m = self.matrix_.shape
        newMatrix = self.majorityMatrix()
        for i in range(m):
            newMatrix[i, i] = n

        return np.argmax(newMatrix.min(axis=1))
