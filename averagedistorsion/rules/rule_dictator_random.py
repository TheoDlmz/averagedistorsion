import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleDictatorRandom(RuleRanking):
    """
    The dictatorship of a random voter
    """

    name = "Random Dictatorship"

    @cached_property
    def ranking_(self):
        return np.argsort(self.matrix_[np.random.randint(self.matrix_.shape[0])])[::-1]

