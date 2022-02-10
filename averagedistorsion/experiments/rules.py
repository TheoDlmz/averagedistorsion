import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class votingRule(DeleteCacheMixin):

    def __init__(self, irrelevant_candidates=0):
        self.irrelevant_candidates = irrelevant_candidates

    def __call__(self, matrix):
        self.delete_cache()
        self.matrix_ = matrix
        return self

    @cached_property
    def ranking_(self):
        raise NotImplementedError

    @cached_property
    def winner_(self):
        for i in self.ranking_:
            if i < self.matrix_.shape[1]-self.irrelevant_candidates:
                return i

    @cached_property
    def utilities_(self):
        return self.matrix_.sum(0)

    @cached_property
    def distortion_(self):
        util = self.utilities_
        if self.irrelevant_candidates == 0:
            return max(1, np.max(util) / util[self.winner_])
        else:
            return max(1, np.max(util[:-self.irrelevant_candidates]) / util[self.winner_])


class plurality(votingRule):

    name = "plurality"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            p = np.argmax(row)
            score[p] += 1

        return np.argsort(score)[::-1]


class veto(votingRule):

    name = "veto"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            p = np.argmin(row)
            score[p] -= 1

        return np.argsort(score)[::-1]


class borda(votingRule):

    name = "borda"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            r = np.argsort(row)
            for i in range(m):
                score[r[i]] += i

        return np.argsort(score)[::-1]


class halfApproval(votingRule):

    name = "half approval"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            r = np.argsort(row)[::-1]
            for i in range(m // 2):
                score[r[i]] += 1

        return np.argsort(score)[::-1]


class harmonic(votingRule):

    name = "harmonic"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            r = np.argsort(row)[::-1]
            for i in range(m):
                score[r[i]] += 1 / (i + 1)

        return np.argsort(score)[::-1]


class stv(votingRule):

    name = "STV"

    @cached_property
    def ranking_(self):
        n, m = self.matrix_.shape
        matrix_copy = self.matrix_.copy()
        ranking = []
        for i in range(m - 1):
            score = np.zeros(m)
            for row in matrix_copy:
                p = np.argmax(row)
                score[p] += 1
            ranking_i = np.argsort(score)
            for elem in ranking_i:
                if elem not in ranking:
                    ranking.append(elem)
                    loser = elem
                    break
            matrix_copy[:, loser] = 0
        for i in range(m):
            if i not in ranking:
                ranking.append(i)

        return ranking[::-1]


class alwaysWorst(votingRule):

    name = "Always worst"

    @cached_property
    def ranking_(self):
        return np.argsort(self.matrix_.sum(axis=0))


class lottery(votingRule):

    name = "random"

    @cached_property
    def ranking_(self):
        ranking = np.arange(self.matrix_.shape[1])
        np.random.shuffle(ranking)
        return ranking


class firstDictator(votingRule):

    name = "first dictator"

    @cached_property
    def ranking_(self):
        return np.argsort(self.matrix_[0])


class randomDictator(votingRule):

    name = "random dictator"

    @cached_property
    def ranking_(self):
        return np.argsort(self.matrix_[np.random.randint(self.matrix_.shape[0])])


class rankedPairs(votingRule):

    name = "ranked pairs"

    def majorityMatrix(self):
        n, m = self.matrix_.shape
        newMatrix = np.zeros((m, m))
        for row in self.matrix_:
            r = np.argsort(row)
            for i in range(m):
                for j in range(i + 1, m):
                    newMatrix[r[j], r[i]] += 1
                    newMatrix[r[i], r[j]] -= 1
        return newMatrix

    @cached_property
    def winner_(self):
        n, m = self.matrix_.shape
        newMatrix = self.majorityMatrix()
        dominate = [[] for _ in range(m)]
        dominated = [[] for _ in range(m)]
        seen = 0
        pairs_matrix = []
        for i in range(m):
            for j in range(m):
                pairs_matrix.append((newMatrix[i, j], i, j))

        pairs_matrix.sort()
        pairs_matrix = pairs_matrix[::-1]

        cont = True
        while cont:
            val, i, j = pairs_matrix[0]
            pairs_matrix = pairs_matrix[1:]

            if i in dominate[j] or j in dominate[i]:
                continue

            if len(dominated[j]) == 0:
                seen += 1

            if seen == m - 1:
                cont = False
            dominate[i].append(j)
            dominated[j].append(i)

            for k in dominated[i]:
                if j not in dominate[k]:
                    dominate[k].append(j)

            for k in dominate[j]:
                if i not in dominated[k]:
                    dominated[k].append(i)

        for i in range(m):
            if len(dominated[i]) == 0:
                return i


class pluralityWithRunoff(votingRule):

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




