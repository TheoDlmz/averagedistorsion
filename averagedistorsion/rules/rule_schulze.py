import numpy as np
from averagedistorsion.rules.rule import Rule
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleSchulze(Rule):
    """
    The Schulze rule
    """

    name = "Schulze"

    @cached_property
    def winner_(self):
        n, m = self.matrix_.shape
        d = self.majorityMatrix()
        p = np.zeros((m, m)) #paths
        for i in range(m):
            for j in range(m):
                if d[i, j] > d[j, i]:
                    p[i, j] = d[i, j]
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    p[j, k] = max(p[j, k], min(p[j, i], p[i, k]))
        for i in range(m):
            winner = True
            for j in range(m):
                if p[j, i] > p[i, j]:
                    winner = False
                    break
            if winner:
                return i
        assert False # there always exists a Schulze winner
