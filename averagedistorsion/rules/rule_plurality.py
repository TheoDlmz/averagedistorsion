import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RulePlurality(RuleRanking):
    """
    The plurality rule, in which every voter give one point to one candidate
    """

    name = "Plurality"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            p = np.argmax(row)
            score[p] += 1

        return np.argsort(score)[::-1]


