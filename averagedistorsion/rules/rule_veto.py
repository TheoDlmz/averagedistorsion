import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleVeto(RuleRanking):
    """
    The veto rule, in which every voter give one point to all but one candidate
    """

    name = "Veto"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            p = np.argmin(row)
            score[p] += 1

        return np.argsort(score)
