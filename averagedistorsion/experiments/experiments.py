import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property
from averagedistorsion.experiments.rules import borda
from averagedistorsion.experiments.models import uniformNormalized


class experimentDistortion(DeleteCacheMixin):

    def __init__(self, rule=None, model=None):
        if model is None:
            model = uniformNormalized()
        self.model = model

        if rule is None:
            rule = borda()
        self.rule = rule

    def election(self, n_voters, n_candidates):
        matrix = self.model(n_voters, n_candidates)
        return self.rule(matrix).distortion_

    def __call__(self, n_voters, n_candidates, n_tries=10000, irrelevant_candidates=0):
        self.delete_cache()
        self.rule.irrelevant_candidates = irrelevant_candidates
        res = []
        for _ in range(n_tries):
            res.append(self.election(n_voters, n_candidates))

        self.results_ = res
        return self

    @cached_property
    def averageDistortion_(self):
        return np.mean(self.results_)

    @cached_property
    def accuracy_(self):
        acc = 0
        for el in self.results_:
            if el == 1:
                acc += 1
        return acc/len(self.results_)
