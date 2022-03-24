import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property


class model(DeleteCacheMixin):

    def __call__(self, n_voters, n_candidates):
        raise NotImplementedError


class uniformNormalized(model):

    def __call__(self, n_voters, n_candidates):
        pos = np.random.rand(n_voters, n_candidates)
        return (pos.T / pos.sum(axis=1)).T


class uniform(model):

    def __call__(self, n_voters, n_candidates):
        return np.random.rand(n_voters, n_candidates)

class euclidean(model):

    def __init__(self, dim=2):
        self.dim = dim

    def generate_points(self, n_points):
        raise NotImplementedError

    def __call__(self, n_voters, n_candidates):
        p_voters = self.generate_points(n_voters)
        p_candidates = self.generate_points(n_candidates)
        result = np.zeros((n_voters, n_candidates))
        for i in range(n_voters):
            for j in range(n_candidates):
                result[i, j] = 1 / np.sqrt(sum((p_voters[i][k] - p_candidates[j][k])**2 for k in range(self.dim)))
        return result

class uniformEuclidean(euclidean):

    def generate_points(self, n_points):
        return np.random.rand(n_points, self.dim)

class gaussianEuclidean(euclidean):

    def __init__(self, loc=0.5, phi=0.2, dim=2):
        self.dim = dim
        self.phi = phi
        self.loc = loc

    def generate_points(self, n_points):
        return np.random.normal(self.loc, self.phi, size=(n_points, self.dim))

class multiplePolesEuclidean(euclidean):

    def __init__(self, poles_num=3, phi=0.2, dim=2):
        self.poles_num = poles_num
        self.dim = dim
        self.phi = phi

    def generate_points(self, n_points):
        poles_points = np.random.rand(self.poles_num, self.dim)
        poles_weights = np.random.rand(self.poles_num)
        poles_weights_sum = poles_weights.sum()
        poles_sizes = [int(x * n_points / poles_weights_sum) for x in poles_weights]
        while sum(poles_sizes) < n_points:
            i  = np.random.randint(self.poles_num)
            poles_sizes[i] += 1
        points = np.zeros((n_points, self.dim))
        start = 0
        for i in range(self.poles_num):
            for j in range(poles_sizes[i]):
                for k in range(self.dim):
                    points[start + j][k] = np.random.normal(loc = poles_points[i][k], scale = self.phi)
            start += poles_sizes[i]
        return points

class identical(model):

    def __init__(self, phi=0):
        self.phi = phi

    def __call__(self, n_voters, n_candidates):
        voter_pref = np.random.rand(n_candidates)
        matrix_id = np.stack([voter_pref for _ in range(n_voters)])
        return (1-self.phi)*matrix_id + self.phi*np.random.rand(n_voters, n_candidates)


class gaussian(model):

    def __init__(self, phi=0):
        self.phi = phi

    def __call__(self, n_voters, n_candidates):
        voter_pref = np.random.rand(n_candidates)
        matrix_id = np.stack([voter_pref for _ in range(n_voters)])
        return matrix_id + np.random.normal(0, self.phi, size=(n_voters, n_candidates))

class gaussianMultimodal(model):

    def __init__(self, phi=0.2, n_peaks=2):
        self.phi = phi
        self.n_peaks = n_peaks

    def __call__(self, n_voters, n_candidates):
        result = np.zeros((n_voters, n_candidates))
        for i in range(self.n_peaks):
            result += np.random.normal(i/(self.n_peaks - 1), self.phi, size=(n_voters, n_candidates))
        return result / self.n_peaks
