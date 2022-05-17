import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property
from averagedistorsion.experiments.experiment import Experiment
from tqdm import tqdm
import matplotlib.pyplot as plt


class ExperimentIrrelevantCandidates(DeleteCacheMixin):
    """
    An experiment in which we increase the number of irrelevant candidates
    """

    def __init__(self, list_rules, n_candidates=2, n_irrelevant_candidates=8, n_voters=20, n_tries=10000):
        self.list_rules = list_rules
        self.n_candidates = n_candidates
        self.n_irrelevant_candidates = n_irrelevant_candidates
        self.n_voters= n_voters
        self.n_tries = n_tries
        self.res = []
        self.accuracy = []

    def __call__(self, model):
        self.res = []
        self.accuracy = []
        for j in tqdm(range(len(self.list_rules))):
            rule = self.list_rules[j]
            elector = Experiment(rule=rule, model=model)
            tab_res = []
            tab_accuracy = []
            for irrelevant_candidate in range(self.n_irrelevant_candidates+1):
                elector(n_voters=self.n_voters, n_candidates=self.n_candidates + irrelevant_candidate,
                        n_tries=self.n_tries, irrelevant_candidates=irrelevant_candidate)
                tab_res.append(elector.averageDistortion_)
                tab_accuracy.append(elector.accuracy_)
            self.res.append(tab_res)
            self.accuracy.append(tab_accuracy)

    def show_distortion(self, titre, show=True):
        plt.figure(figsize=(20, 10))
        style = '-'
        for j in range(len(self.list_rules)):
            if j >= 10:
                style = '--'
            plt.plot(range(self.n_irrelevant_candidates+1), self.res[j], style, label=self.list_rules[j].name, linewidth=2)
        plt.legend()
        plt.title("%s, m=%i, n=%i"%(titre, self.n_candidates, self.n_voters))
        plt.xlabel("Irrelevant candidates")
        plt.ylabel("Average distortion")
        plt.ylim(1)
        plt.xlim(0, self.n_irrelevant_candidates)
        if show:
            plt.show()

    def show_accuracy(self, titre, show=True):
        plt.figure(figsize=(20, 10))
        style = '-'
        for j in range(len(self.list_rules)):
            if j >= 10:
                style = '--'
            plt.plot(range(self.n_irrelevant_candidates+1), self.accuracy[j], style, label=self.list_rules[j].name, linewidth=2)
        plt.legend()
        plt.xlim(0, self.n_irrelevant_candidates)
        plt.ylim(0, 1)
        plt.title("%s, m=%i, n=%i"%(titre, self.n_candidates, self.n_voters))
        plt.xlabel("Irrelevant candidates")
        plt.ylabel("Accuracy")
        if show:
            plt.show()


# Save & Load results
