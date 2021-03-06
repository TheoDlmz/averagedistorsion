import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleHalfApproval(RuleRanking):
    """
    The approval rule in which half of the candidates are approved
    """

    name = "Half approval"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            r = np.argsort(row)[::-1]
            for i in range(m // 2):
                score[r[i]] += 1

        return np.argsort(score)[::-1]
