import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleBaldwin(RuleRanking):
    """
    The Baldwin rule
    """

    name = "Baldwin"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        matrix_copy = self.matrix_.copy()
        ranking = []
        for i in range(m - 1):
            score = np.zeros(m)
            for row in matrix_copy:
                rank = np.argsort(row)[::-1]
                points = m - len(ranking) - 1
                for p in rank:
                    if p not in ranking:
                        score[p] += points
                        points -= 1
                assert points == -1
            ranking_i = np.argsort(score)
            for elem in ranking_i:
                if elem not in ranking:
                    ranking.append(elem)
                    loser = elem
                    break
            matrix_copy[:, loser] = - np.infty
        for i in range(m):
            if i not in ranking:
                ranking.append(i)

        return ranking[::-1]
