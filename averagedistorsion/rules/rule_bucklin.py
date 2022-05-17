import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleBucklin(RuleRanking):
    """
    The Bucklin rule
    """

    name = "Bucklin"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        ranking = []
        for k in range(m):
            unique, counts = np.unique(np.argsort(-self.matrix_, axis=1)[:, :k], return_counts=True)
            val = []
            for i in range(len(unique)):
                if unique[i] not in ranking and counts[i] >= n/2:
                    val.append((counts[i], unique[i]))

            val = sorted(val)[::-1]
            ranking.extend([el for _, el in val])
        return ranking
