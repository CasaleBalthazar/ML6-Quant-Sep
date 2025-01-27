"""
Contain real-valued vector representation for density matrices
"""

from ..types import DMStack

import numpy as np


class GellMann :
    """
    The decomposition into the generalised Gell-Mann basis
    """
    @staticmethod
    def representation(states, infos={}):
        """
        return the representation associated to each state
        :param states: a set of density matrices
        :param infos: a dictionary of information
        :return: a set of real-valued vectors (DMStack), a dictionary of information (empty)
        """
        dim = np.product(states.dims)
        vecs = np.zeros((len(states), dim**2-1))

        idx = 0
        # symmetric components
        for j in range(dim):
            for k in range(j + 1, dim):
                vecs[:, idx] = (states[:, j, k] + states[:, k, j]).real
                idx += 1
        # antisymmetric components
        for j in range(dim):
            for k in range(j + 1, dim):
                vecs[:, idx] = (states[:, j, k] - states[:, k, j]).imag
                idx += 1
        # diagonal components
        for l in range(1, dim):
            vecs[:, idx] = np.sqrt(2 / (l * (l + 1))) * (
                    np.sum(np.diagonal(states, axis1=1, axis2=2)[:, :(l - 1)], axis=1).real - states[:, l, l].real
            )
            idx += 1
        return DMStack(vecs, states.dims), {}


class Measures :
    """
    The representation by the average value of a set of measures
    """
    def __init__(self, observables):
        """
        :param observables: a set of measures
        """
        self.obs = observables

    def representation(self, states, infos={}):
        """
        return the representation associated to each state
        :param states: a set of density matrices
        :param infos: a dictionary of information
        :return: a set of real-valued vectors (DMStack), a dictionary of information (empty)
        """
        vecs = np.zeros((len(states), len(self.obs)))
        for i in range(len(self.obs)) :
            vecs[:,i] = np.trace(np.matmul(self.obs[i], states), axis1=1, axis2=2).real
        return DMStack(vecs, states.dims), {}
