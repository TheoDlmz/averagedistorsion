import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleDictatorFirst(RuleRanking):
    """
    The dictatorship of the first voter
    """

    name = "Dictatorship of 1st"

    @cached_property
    def ranking_(self):
        return np.argsort(self.matrix_[0])[::-1]
