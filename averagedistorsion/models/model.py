import numpy as np
from averagedistorsion.utils.cached import DeleteCacheMixin, cached_property


class Model(DeleteCacheMixin):
    """
    The Model family of Classes is used for Utilities / Distance generation models

    """

    positive = True

    def __call__(self, n_voters, n_candidates):
        """

        Parameters
        ----------
        n_voters: Int
            The number of voters in the model
        n_candidates : Int
            The number of candidates in the model

        Returns
        -------
        np.array
            The utilities of voters for candidates

        """
        raise NotImplementedError
