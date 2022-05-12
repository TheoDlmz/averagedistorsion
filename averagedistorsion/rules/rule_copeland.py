import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleCopeland(RuleRanking):

    name = "Copeland"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        newMatrix = self.majorityMatrix()
        scores = [0]*m
        for i in range(m):
            for j in range(i+1, m):
                if newMatrix[i, j] > 0:
                    scores[i] += 1
                elif newMatrix[i, j] < 0:
                    scores[j] += 1
        return np.argsort(scores)[::-1]
