import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleEgalitarian(RuleRanking):

    name = "egalitarian"

    @cached_property
    def ranking_(self):
        return np.argsort(np.min(self.matrix_, axis=0))[::-1]
