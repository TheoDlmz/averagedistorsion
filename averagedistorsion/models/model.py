import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property


class Model(DeleteCacheMixin):

    def __call__(self, n_voters, n_candidates):
        raise NotImplementedError
