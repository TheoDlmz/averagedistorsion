import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleBorda(RuleRanking):
    """
    The borda rule, that gives m-1 points to the first candidate, m-2 to the second, etc. and 0 to the last one
    """

    name = "Borda"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            r = np.argsort(row)
            for i in range(m):
                score[r[i]] += i

        return np.argsort(score)[::-1]
