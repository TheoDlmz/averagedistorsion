import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleRandomVoterUtility(RuleRanking):
    """
    The rule that use the utility vector of the first voter as a positional scoring rule
    """

    name = "Random Voter Utility"

    @cached_property
    def ranking_(self):
        positional_rule = np.sort(self.matrix_[0])
        n, m = self.matrix_.shape
        score = np.zeros(m)
        for row in self.matrix_:
            r = np.argsort(row)
            for i in range(m):
                score[r[i]] += positional_rule[i]

        return np.argsort(score)[::-1]
