import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleDictatorRandom(RuleRanking):

    name = "random dictator"

    @cached_property
    def ranking_(self):
        return np.argsort(self.matrix_[np.random.randint(self.matrix_.shape[0])])[::-1]

