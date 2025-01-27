"""
Contain methods to obtain the separable approximation of a quantum state
"""

from ..types  import DMStack
from ..samplers.pure import RandomHaar
from ..samplers.utils import kron, dagger

import numpy as np


class FrankWolfe:
    """
    Adaptation of the Frank Wolfe algorithm to the separability problem
    """
    def __init__(self, n_steps, return_dists=False):
        """
        :param n_steps: number of steps
        :param return_dists: if true, return the distance separating the state to its approximation at each step
                             as information (key : 'dists')
        """
        self.n_steps = n_steps
        self.return_dists = return_dists

    def approximation(self, states, infos={}):
        """
        return an approximation to the closest separable state
        :param states: density matrices
        :param infos: dictionary of information (unused)
        :return: a set of quantum states (DMStack), a dictionary of information
        """

        def max_eigvecs(Ms):
            _, eigvecs = np.linalg.eigh(Ms, UPLO='U')
            return eigvecs[:, :, -1]

        def max_schmidts(dims, vecs):
            mats = vecs.reshape((len(vecs), dims[0], dims[1]), order='F')
            u_mat, sing_vals, vt_mat = np.linalg.svd(mats)
            vt_mat = np.transpose(vt_mat, axes=(0, 2, 1))

            return vt_mat[:, :, 0], u_mat[:, :, 0]

        dims = states.dims
        dim = np.product(dims)
        approx = RandomHaar(product=True).states(len(states), dims)[0]

        if self.return_dists :
            dists = np.zeros((len(states), self.n_steps))

        for t in range(self.n_steps):
            if self.return_dists :
                dists[:,t] = np.linalg.norm(approx - states, axis=(1,2))

            v = max_eigvecs(states - approx)
            s1, s2 = max_schmidts(dims, v)
            # from vector to density matrix
            S = kron(s1,s2)
            S = S.reshape((len(states), dim, 1))
            S = S @ dagger(S)

            step_size = 2 / (t + 2)
            approx = (1 - step_size) * approx + step_size * S

        if self.return_dists :
            return DMStack(approx, dims), {'dists' : dists}
        else :
            return DMStack(approx, dims), {}

    def witness(self, states, infos={}):
        """
        return an approximation of the optimal entanglement witness
        :param states: a set of density matrices
        :param infos: a dictionary of information (unused)
        :return: a set of witnesses (
        """
        dims = states.dims
        dim = np.product(states.dims)

        approx, infos = self.approximation(states, infos)
        C = np.trace(approx @ (approx - states), axis1=1, axis2=2)

        wit = approx - states - C[:,None,None] * np.full(states.shape, np.eye(dim))
        wit /= np.trace(wit, axis1=1,axis2=2)[:, None, None]

        return DMStack(wit, dims), infos
