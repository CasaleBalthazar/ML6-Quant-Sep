"""
Contain methods to sample entangled density matrices
"""

import numpy as np

from ..types import DMStack
from .utils import balanced, RandomUnitary, dagger

class AugmentedPPTEnt :
    """
    Random PPT-ENT states obtained via data-augmentation of an initial set of PPT-ENT states
    """
    def __init__(self, states, infos, wit_key, noise_mthd):
        """
        :param states: a set of PPT-ENT states
        :param infos: a dictionary of information
        :param wit_key: the key containing the optimal witness in the dictionary
        :param noise_mthd: a sampler method to produce noise
        """
        self.seeds = states
        self.wits = infos[wit_key]
        self.noise_mthd = noise_mthd

        # compute all necessary values at initialisation.
        eigvals = np.linalg.eigvalsh(self.wits)
        self.Ls = -np.trace(self.wits @ self.seeds, axis1=1, axis2=2).real
        self.n_pos = np.sum(eigvals > 0, axis=1) # n positive eigenvalues
        self.s_pos = np.sum((eigvals > 0) * eigvals, axis=1) # sum positive eigenvalues


    def states(self, n_states, dims=None):
        """
        sample n_states PPT-ENT density matrices using data-augmentation
        :param n_states: number of density matrices
        :param dims: dimensions of the subsystems (unused)
        :return: a set of states (DMStack), a dictionary of information (empty)
        """
        dim = np.product(self.seeds.dims)
        dims = self.seeds.dims
        n_per_seed = balanced(n_states, len(self.seeds))


        states = np.zeros((n_states, dim, dim), dtype=complex)
        idx = 0
        for i in range(len(n_per_seed)) :
            if n_per_seed[i] == 0 :
                continue
            states[idx : idx + n_per_seed[i]] = self.states_seed(i, n_per_seed[i], dims)[0]
            idx += n_per_seed[i]

        return DMStack(states, self.seeds.dims), {}

    def states_seed(self, idx, n_states, dims):
        dim = np.product(dims)

        def get_xys() :
            xs = np.random.uniform(1 / (1 + dim*self.Ls[idx]), 1, size=n_states)
            y1s = (1 - xs) / (dim - 1 - xs)
            y2s = self.n_pos[idx] * (xs * (1 + dim*self.Ls[idx]) - 1)
            y2s /= (dim * self.s_pos[idx] + y2s)

            ys = np.minimum(y1s, y2s) # max dispersion
            return xs, ys

        states = np.full((n_states, dim, dim), self.seeds[idx])
        noises = self.noise_mthd(n_states, dims)[0]

        # step 1 : sample in the volume
        xs, ys = get_xys()
        states = xs[:,None,None] * states + (1 - xs)[:, None, None] * np.full(states.shape, np.eye(dim)/dim)
        states = ys[:,None,None] * noises + (1 - ys)[:, None, None] * states

        # step 2 : apply random rotations
        U = RandomUnitary(product=True).matrices(n_states, dims)
        states = U @ states @ dagger(U)

        return DMStack(states, self.seeds.dims), {}
