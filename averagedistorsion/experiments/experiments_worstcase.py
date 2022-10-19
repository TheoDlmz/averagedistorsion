import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property
from averagedistorsion.models.model_uniform_normalized import ModelUniformNormalized
from tqdm import tqdm
import matplotlib.pyplot as plt


class ExperimentWorstCase(DeleteCacheMixin):
    """
    An experiment in which we increase the number of irrelevant candidate
    """

    def __init__(self, list_rules, n_candidates=2, n_irrelevant_candidates=8,  n_tries=10000):
        self.list_rules = list_rules
        self.n_candidates = n_candidates
        self.n_irrelevant_candidates = n_irrelevant_candidates
        self.res = []
        self.n_tries = n_tries

    def compute_dist(self,irrelevant_candidate, dist, r):
        m_r = 2
        m_i = irrelevant_candidate
        m = m_i + m_r
        d = [x / 2 for x in dist]

        if r == "borda":
            r_vec = list(np.linspace(1, 0, m))  # Borda
        elif r == "oddBorda":
            r_vec = [(2 * (m_i - i) + 1) / (2 * m_i + 1) for i in range(m_i + 1)] + [0]  # Scoring rule used
        elif r == "harmonic":
            r_vec = [(1 / i - 1 / m) * (m / (m - 1)) for i in range(1, m + 1)]
        elif r == "kApp":
            k = int(m/2)
            r_vec = [1] * k + [0] * (m - k)
        elif r == "plurality":
            r_vec = [1] * 1 + [0] * (m - 1)
        elif r == "veto":
            r_vec = [1] * (m-1) + [0] * 1

        max_dist = 0
        for i in range(m - 1):
            for j in range(m - 1):
                if r_vec[j] == 0:
                    c = 1 - d[j]
                else:
                    c = (r_vec[j] * d[i + 1] + r_vec[i] * (1 - d[j])) / (r_vec[i] + r_vec[j])
                max_dist = max(c, max_dist)

        return max_dist / (1 - max_dist)

    def __call__(self, model):
        self.res = []
        for j in tqdm(range(len(self.list_rules))):
            rule = self.list_rules[j]
            tab_res = []
            for irrelevant_candidate in range(self.n_irrelevant_candidates+1):
                max_dist = 0
                for i in range(self.n_tries):
                    pos = sorted(list(model(irrelevant_candidate)))
                    pos = [0] + pos + [1]
                    max_dist_1 = self.compute_dist(irrelevant_candidate, pos, rule)
                    max_dist_2 = self.compute_dist(irrelevant_candidate, [1-p for p in pos[::-1]], rule)
                    max_dist += max(max_dist_1, max_dist_2)
                tab_res.append(max_dist/self.n_tries)

            self.res.append(tab_res)

    def show_distortion(self, titre, show=True):
        plt.figure(figsize=(20, 10))
        style = 'o-'
        for j in range(len(self.list_rules)):
            if j >= 10:
                style = 'o--'
            plt.plot(range(self.n_irrelevant_candidates+1), self.res[j], style, label=self.list_rules[j], linewidth=2)
        plt.legend()
        plt.title("%s, m=%i"%(titre, self.n_candidates))
        plt.xlabel("Irrelevant candidates")
        plt.ylabel("Average worst distortion")
        plt.ylim(1)
        plt.xlim(0, self.n_irrelevant_candidates)
        if show:
            plt.show()

    def save_results(self, filename):
        columns = ["rule", "n_relevant_candidates", "n_irrelevant_candidates",
                   "n_tries", "distortion"]
        out = []
        for j in range(len(self.list_rules)):
            for i in range(self.n_irrelevant_candidates+1):
                out.append([self.list_rules[j], self.n_candidates, i,
                            self.n_tries, self.res[j][i]])

        np.savetxt(filename, np.array(out), fmt="%s", delimiter=",", header=",".join(columns))



# Save & Load results
