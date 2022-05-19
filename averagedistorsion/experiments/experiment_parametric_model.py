import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property
from averagedistorsion.experiments.experiment import Experiment
from tqdm import tqdm
import matplotlib.pyplot as plt


class ExperimentParametricModel(DeleteCacheMixin):
    """
    An experiment in which we increase the number of voters
    """
    def __init__(self, list_rules, n_relevant_candidates=10, n_irrelevant_candidates=0, n_voters=20, n_tries=10000):
        if n_voters is None:
            n_voters = [10*(i+1) for i in range(10)]
        self.list_rules = list_rules
        self.n_relevant_candidates = n_relevant_candidates
        self.n_irrelevant_candidates = n_irrelevant_candidates
        self.n_voters = n_voters
        self.n_tries = n_tries
        self.res = []
        self.accuracy = []
        self.parameter_list = []

    def __call__(self, parametric_model, parameter_list):
        self.res = []
        self.accuracy = []
        self.parameter_list = parameter_list
        for j in tqdm(range(len(self.list_rules))):
            rule = self.list_rules[j]
            tab_res = []
            tab_accuracy = []
            for parameter in self.parameter_list:
                elector = Experiment(rule=rule, model=parametric_model(parameter))
                elector(n_voters=self.n_voters, n_candidates=self.n_relevant_candidates+self.n_irrelevant_candidates,
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
            plt.plot(self.parameter_list, self.res[j], style, label=self.list_rules[j].name, linewidth=2)
        plt.legend()
        plt.title("%s, n=%i, m=%i, m'=%i"%(titre, self.n_voters, self.n_relevant_candidates,
                                           self.n_irrelevant_candidates))
        plt.xlabel("Param")
        plt.ylabel("Average distortion")
        plt.ylim(1)
        plt.xlim(self.parameter_list[0], self.parameter_list[-1])
        if show:
            plt.show()

    def show_accuracy(self, titre, show=True):
        plt.figure(figsize=(20, 10))
        style = '-'
        for j in range(len(self.list_rules)):
            if j >= 10:
                style = '--'
            plt.plot(self.parameter_list, self.accuracy[j], style, label=self.list_rules[j].name, linewidth=2)
        plt.legend()
        plt.xlim(self.parameter_list[0], self.parameter_list[-1])
        plt.ylim(0, 1)
        plt.title("%s, n=%i, m=%i, m'=%i"%(titre, self.n_voters, self.n_relevant_candidates,
                                           self.n_irrelevant_candidates))
        plt.xlabel("Param")
        plt.ylabel("Accuracy")
        if show:
            plt.show()

    def save_results(self, filename):
        columns = ["rule", "n_voters", "n_relevant_candidates", "n_irrelevant_candidates",
                   "n_tries", "parameter", "distortion", "accuracy"]
        out = []
        for j in range(len(self.list_rules)):
            for i, v in enumerate(self.parameter_list):
                out.append([self.list_rules[j].name, self.n_voters, self.n_relevant_candidates,
                            self.n_irrelevant_candidates, self.n_tries, v,
                            self.res[j][i], self.accuracy[j][i]])

        np.savetxt(filename, np.array(out), fmt="%s", delimiter=",", header=",".join(columns))

