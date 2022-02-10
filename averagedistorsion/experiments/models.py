import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property


class model(DeleteCacheMixin):

    def __call__(self, n_voters, n_candidates):
        raise NotImplementedError


class uniformNormalized(model):

    def __call__(self, n_voters, n_candidates):
        pos = np.random.rand(n_voters, n_candidates)
        return (pos.T / pos.sum(axis=1)).T
