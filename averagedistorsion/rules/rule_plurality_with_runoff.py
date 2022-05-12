import numpy as np
from averagedistorsion.rules.rule import Rule
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RulePluralityWithRunoff(Rule):

    name = "plurality w/ runoff"

    @cached_property
    def winner_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            p = np.argmax(row)
            score[p] += 1

        [c1, c2] = np.argsort(score)[-2:]
        score_runoff = 0
        for row in self.matrix_:
            if row[c1] > row[c2]:
                score_runoff += 1
            else:
                score_runoff -= 1
        if score_runoff >= 0:
            return c1
        return c2
