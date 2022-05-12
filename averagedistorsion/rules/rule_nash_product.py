import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleNashProduct(RuleRanking):

    name = "Nash Product"

    @cached_property
    def ranking_(self):
        return np.argsort(np.product(self.matrix_, axis=0))[::-1]
