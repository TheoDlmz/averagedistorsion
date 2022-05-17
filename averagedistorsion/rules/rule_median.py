import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleMedian(RuleRanking):
    """
    The median rule, i.e. the rule that select the candidate with the best median utility
    """

    name = "Median"

    @cached_property
    def ranking_(self):
        return np.argsort(np.median(self.matrix_, axis=0))[::-1]
