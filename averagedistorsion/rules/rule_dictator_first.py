import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleDictatorFirst(RuleRanking):

    name = "first dictator"

    @cached_property
    def ranking_(self):
        return np.argsort(self.matrix_[0])[::-1]
