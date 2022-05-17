import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleAlwaysWorst(RuleRanking):
    """
    A rule that always select the worst possible candidate
    """

    name = "Always Worst"

    @cached_property
    def ranking_(self):
        return np.argsort(self.matrix_.sum(axis=0))
