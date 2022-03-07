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
