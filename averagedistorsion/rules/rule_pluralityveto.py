import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RulePluralityVeto(RuleRanking):
    """
    The pluralityVeto rule, in which every voter give one point to one candidate such that every voter
    has a plurality score, then voters are sequentially picked in a random order and they remove one point
    to their least favorite voter that still have a positive score. The last remaining candidate is the winner.
    """

    name = "PluralityVeto"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            p = np.argmax(row)
            score[p] += 1

        voters_order = np.arange(n)
        np.random.shuffle(voters_order)
        ranking = []
        for i in range(m):
            if score[i] == 0:
                ranking.append(i)
        for i in voters_order:
            order_i = np.argsort(self.matrix_[i])
            for j in range(m):
                candidate_j = order_i[j]
                if score[candidate_j] > 0:
                    score[candidate_j] -= 1
                    if score[candidate_j] == 0:
                        ranking.append(candidate_j)
                    break

        return np.array(ranking)[::-1]



