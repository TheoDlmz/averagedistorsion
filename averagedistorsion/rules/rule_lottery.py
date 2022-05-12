import numpy as np
from averagedistorsion.rules.rule_ranking import RuleRanking
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class RuleLottery(RuleRanking):

    name = "lottery"

    @cached_property
    def ranking_(self):
        ranking = np.arange(self.matrix_.shape[1])
        np.random.shuffle(ranking)
        return ranking
