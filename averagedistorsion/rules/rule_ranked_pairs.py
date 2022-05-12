import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property


class RuleRankedPairs(RuleRanking):

    name = "Ranked Pairs"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        newMatrix = self.majorityMatrix()
        dominate = [[] for _ in range(m)]
        dominated = [[] for _ in range(m)]
        seen = 0
        pairs_matrix = []
        for i in range(m):
            for j in range(m):
                if i != j:
                    pairs_matrix.append((newMatrix[i, j], i, j))

        pairs_matrix.sort()
        pairs_matrix = pairs_matrix[::-1]

        while True:
            val, i, j = pairs_matrix[0]
            if val <= 0:
                break

            pairs_matrix = pairs_matrix[1:]

            if i in dominated[j] or j in dominate[i]:
                continue

            if len(dominated[j]) == 0:
                seen += 1

            if j in dominated[i] or i in dominate[j]:
                continue

            dominate[i].append(j)
            dominated[j].append(i)

            for k in dominated[i]:
                if j not in dominate[k]:
                    dominate[k].append(j)

            for k in dominate[j]:
                if i not in dominated[k]:
                    dominated[k].append(i)

        ranking = []
        dominated_count = []
        for i in range(m):
            dominated_count.append(len(dominated[i]))

        for _ in range(m):
            selected = -1
            for i in range(m):
                if (dominated_count[i] == 0) and (i not in ranking):
                    selected = i
            ranking.append(selected)
            for i in range(m):
                if selected in dominated[i]:
                    dominated_count[i] -= 1

        return ranking
