import numpy as np
from averagedistorsion.rules.rule import Rule
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleMaximin(Rule):
    """
    The Maximin rule, in which the winner is the candidate that maximizes its score in its worst duel against another
    candidate
    """

    name = "Maximin"

    @cached_property
    def winner_(self):
        n, m = self.matrix_.shape
        newMatrix = self.majorityMatrix()
        for i in range(m):
            newMatrix[i, i] = n

        return np.argmax(newMatrix.min(axis=1))
