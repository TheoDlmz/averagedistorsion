import matplotlib.pyplot as plt
import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin,cached_property


class DistortionTwoCand(DeleteCacheMixin):
    def __init__(self, n_voters=20):
        self.n_voters = n_voters
        self.results_ = None

    def generateScores(self):
        pos = np.random.rand(self.n_voters)
        dist_a = pos.sum()
        score_a = (pos >= 0.5).sum()
        return dist_a, score_a

    def computeDistortion(self):
        raise NotImplemented

    def __call__(self, n_tries=100000):
        self.delete_cache()
        tab_exp = []
        for i in range(n_tries):
            tab_exp.append(self.computeDistortion())
        self.results_ = tab_exp
        return self

    def showDistribution(self):
        plt.figure(figsize=(10, 5))
        plt.yscale("log")
        plt.hist(self.results_, bins=25)
        plt.xlabel("distortion")
        plt.title("Distribution of the distortion")
        plt.ylabel("count")
        plt.show()

    @cached_property
    def averageDistortion_(self):
        return np.mean(self.results_)


class adversarialDistortionTwoCand(DistortionTwoCand):

    def computeDistortion(self):
        dist_a, score_a = self.generateScores()
        if score_a == self.n_voters-score_a:
            return max(dist_a/(self.n_voters-dist_a), (self.n_voters-dist_a)/dist_a)
        elif score_a > self.n_voters-score_a:
            return max(1, (self.n_voters-dist_a)/dist_a)
        else:
            return max(1, dist_a/(self.n_voters-dist_a))


class antiAdversarialDistortionTwoCand(DistortionTwoCand):

    def computeDistortion(self):
        dist_a, score_a = self.generateScores()
        if score_a == self.n_voters-score_a:
            return 1
        elif score_a > self.n_voters-score_a:
            return max(1, (self.n_voters-dist_a) / dist_a)
        else:
            return max(1, dist_a / (self.n_voters-dist_a))


class consistentDistortionTwoCand(DistortionTwoCand):

    def computeDistortion(self):
        dist_a, score_a = self.generateScores()
        if score_a >= self.n_voters-score_a:
            return max(1, (self.n_voters-dist_a)/dist_a)
        else:
            return max(1, dist_a/(self.n_voters-dist_a))
