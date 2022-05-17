import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property
from averagedistorsion.rules.rule import Rule


class RuleRanking(Rule):
    """
    The family of rules that actually output a ranking of the candidates

    Parameters
    ----------
    irrelevant_candidates: int
        The number of candidates that cannot win the election.
    """

    def __init__(self, irrelevant_candidates=0):
        self.irrelevant_candidates = irrelevant_candidates

    @cached_property
    def ranking_(self):
        raise NotImplementedError

    @cached_property
    def winner_(self):
        for i in self.ranking_:
            if i < self.matrix_.shape[1]-self.irrelevant_candidates:
                return i

    @cached_property
    def distortion_(self):
        util = self.utilities_
        if self.irrelevant_candidates == 0:
            return max(1, np.max(util) / util[self.winner_])
        else:
            return max(1, np.max(util[:-self.irrelevant_candidates]) / util[self.winner_])

    @cached_property
    def cost_(self):
        util = -self.utilities_
        if self.irrelevant_candidates == 0:
            return max(1, util[self.winner_]/np.min(util))
        else:
            return max(1, util[self.winner_]/np.min(util[:-self.irrelevant_candidates]))

