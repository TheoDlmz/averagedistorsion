import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleEgalitarian(RuleRanking):
    """
    The rule that directly use utilities to select the egalitarian winner, i.e. the one that maximize the
    minimum utility
    """

    name = "Egalitarian"

    @cached_property
    def ranking_(self):
        return np.argsort(np.min(self.matrix_, axis=0))[::-1]
