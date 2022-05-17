import numpy as np
from averagedistorsion.models.model import Model


class ModelFromDataset(Model):
    """
    A model in which utilities are drawn from a utility matrix

    Parameters
    ----------
    dataset: np.array
        The matrix of utilities for every voters
    noise: float
        The std of the noise that we want to add to the data
    """

    def __init__(self, dataset, noise=0):
        dataset = np.array(dataset)
        self.dataset = dataset
        self.noise = noise
        self.n_voters, self.n_candidates = dataset.shape

    def __call__(self, n_voters, n_candidates):
        if n_voters > self.n_voters:
            raise ValueError("too many voters")
        if n_candidates > self.n_candidates:
            raise ValueError("too many candidates")

        list_voters = np.arange(self.n_voters)
        np.random.shuffle(list_voters)
        voters = list_voters[:n_voters]

        list_candidates = np.arange(self.n_candidates)
        np.random.shuffle(list_candidates)
        candidates = list_candidates[:n_candidates]

        return self.dataset[voters][:, candidates] \
               + self.noise*np.random.normal(size=(n_voters, n_candidates))

