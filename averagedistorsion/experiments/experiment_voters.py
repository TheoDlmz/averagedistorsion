import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property
from averagedistorsion.experiments.experiment import Experiment
from tqdm import tqdm
import matplotlib.pyplot as plt


class ExperimentVoters(DeleteCacheMixin):
    """
    An experiment in which we increase the number of voters
    """
    def __init__(self, list_rules, n_relevant_candidates=10, n_irrelevant_candidates=0, n_voters=None, n_tries=10000):
        if n_voters is None:
            n_voters = [10*(i+1) for i in range(10)]
        self.list_rules = list_rules
        self.n_relevant_candidates = n_relevant_candidates
        self.n_irrelevant_candidates = n_irrelevant_candidates
        self.n_voters = n_voters
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
            for voter in self.n_voters:
                elector(n_voters=voter, n_candidates=self.n_relevant_candidates+self.n_irrelevant_candidates,
                        n_tries=self.n_tries, irrelevant_candidates=self.n_irrelevant_candidates)
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
            plt.plot(self.n_voters, self.res[j], style, label=self.list_rules[j].name, linewidth=2)
        plt.legend()
        plt.title("%s, m=%i, m'=%i"%(titre, self.n_relevant_candidates, self.n_irrelevant_candidates))
        plt.xlabel("Voters")
        plt.ylabel("Average distortion")
        plt.ylim(1)
        plt.xlim(self.n_voters[0], self.n_voters[-1])
        if show:
            plt.show()

    def show_accuracy(self, titre, show=True):
        plt.figure(figsize=(20, 10))
        style = '-'
        for j in range(len(self.list_rules)):
            if j >= 10:
                style = '--'
            plt.plot(self.n_voters, self.accuracy[j], style, label=self.list_rules[j].name, linewidth=2)
        plt.legend()
        plt.xlim(self.n_voters[0], self.n_voters[-1])
        plt.ylim(0, 1)
        plt.title("%s, m=%i, m'=%i"%(titre, self.n_relevant_candidates, self.n_irrelevant_candidates))
        plt.xlabel("Voters")
        plt.ylabel("Accuracy")
        if show:
            plt.show()

    def save_results(self, filename):
        columns = ["rule", "n_voters", "n_relevant_candidates", "n_irrelevant_candidates",
                   "n_tries", "distortion", "accuracy"]
        out = []
        for j in range(len(self.list_rules)):
            for i, v in enumerate(self.n_voters):
                out.append([self.list_rules[j].name, v, self.n_relevant_candidates, self.n_irrelevant_candidates,
                            self.n_tries, self.res[j][i], self.accuracy[j][i]])

        np.savetxt(filename, np.array(out), fmt="%s", delimiter=",", header=",".join(columns))


# Save & Load results
