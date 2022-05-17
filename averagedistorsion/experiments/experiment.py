import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property
from averagedistorsion.rules.rule_borda import RuleBorda
from averagedistorsion.models.model_uniform_normalized import ModelUniformNormalized


class Experiment(DeleteCacheMixin):
    """
    A class to repeat an experiment with a given rule and given model

    Parameters
    ----------
    rule: Rule
        The preference rule that is used. Default is RuleBorda
    model: Model
        The utility model that is used. Default is ModelUniformNormalized

    """

    def __init__(self, rule=None, model=None):
        if model is None:
            model = ModelUniformNormalized()
        self.model = model
        self.cost = not model.positive

        if rule is None:
            rule = RuleBorda()
        self.rule = rule

    def election(self, n_voters, n_candidates):
        """
        Run an election with n_voters and n_candidates and return the obtained distortion

        Parameters
        ----------
        n_voters: int
            The number of voters
        n_candidates: int
            The number of candidates

        Returns
        -------

        """
        matrix = self.model(n_voters, n_candidates)
        if self.cost:
            return self.rule(matrix).cost_
        else:
            return self.rule(matrix).distortion_

    def __call__(self, n_voters, n_candidates, n_tries=10000, irrelevant_candidates=0):
        """
        Repeat the election process

        Parameters
        ----------
        n_voters: int
            The number of voters in elections
        n_candidates: int
            The number of candidates in elections
        n_tries: int
            The number of elections
        irrelevant_candidates: int
            The number of irrelevant candidates

        Returns
        -------
        np.array
            The array of the distortion for all the iterations

        """
        self.delete_cache()
        self.rule.irrelevant_candidates = irrelevant_candidates
        res = []
        for _ in range(n_tries):
            res.append(self.election(n_voters, n_candidates))

        self.results_ = res
        return self

    @cached_property
    def averageDistortion_(self):
        """
        Returns
        -------
        float
            The average distortion

        """
        return np.mean(self.results_)

    @cached_property
    def accuracy_(self):
        """
        Returns
        -------
        float
            The percentage of accuracy (i.e. the percentage of time we get a distortion of 1)

        """
        acc = 0
        for el in self.results_:
            if el == 1:
                acc += 1
        return acc/len(self.results_)
