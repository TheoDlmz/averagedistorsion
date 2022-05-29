import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleCondorcet(RuleRanking):
    """
    Returns the Condorcet winner if one exists; else raises error
    """

    name = "Condorcet"

    @cached_property
    def winner_(self):
        n, m = self.matrix_.shape
        newMatrix = self.majorityMatrix()
        worst_duels = newMatrix.min(axis=1)
        if np.max(worst_duels) < 0:
            raise Exception("No Condorcet winner!")
        else:
            return np.argmax(worst_duels)
