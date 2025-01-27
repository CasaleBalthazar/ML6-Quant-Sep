"""
Contain methods to sample mixed density matrices
"""

import numpy as np

from ..types import DMStack
from .utils import RandomGinibre, RandomUnitary, dagger, balanced

class RandomInduced :
    """
    Random mixed states from the Induced measure
    """
    def __init__(self, k_params):
        """
        :param k_params: a int or an array of int
        """
        self.k_params = k_params

    def states(self, n_states, dims):
        """
        sample n_states density matrices from the Induced measure
        :param n_states: number of density matrices
        :param dims: dimensions of the subsystems
        :return: a set of states (DMStack), a dictionary of information (k param : key 'ks')
        """
        if isinstance(self.k_params, int):
            return RandomInduced.states_k(self.k_params, n_states, dims)
        else :
            return RandomInduced.states_ks(self.k_params, n_states, dims)

    @staticmethod
    def states_k(k, n_states, dims):
        dim = np.product(dims)
        gin = RandomGinibre.matrices(n_states, [dim, k])
        gin = gin @ dagger(gin)
        gin /= np.trace(gin, axis1=1, axis2=2)[:,None,None]

        return gin, {'ks' : np.full(n_states, k)}


    @staticmethod
    def states_ks(vals_k, n_states, dims):
        dim = np.product(dims)

        per_k = balanced(n_states, len(vals_k))
        states = np.zeros((n_states, dim, dim), dtype=complex)
        ks = np.zeros(n_states, dtype=int)
        idx = 0
        for i in range(len(per_k)):
            if per_k[i]==0 :
                continue
            states[idx : idx + per_k[i]] = RandomInduced.states_k(vals_k[i], per_k[i], dims)[0]
            ks[idx : idx + per_k[i]] = vals_k[i]
            idx += per_k[i]

        return DMStack(states, dims), {'ks' : ks}


class RandomBures :
    """
    Random mixed states from the Bures measure
    """

    @staticmethod
    def states(n_states, dims):
        """
        sample n_states density matrices from the Bures measure
        :param n_states: number of density matrices
        :param dims: dimensions of the subsystems
        :return: a set of states (DMStack), a dictionary of information (empty)
        """
        dim = np.product(dims)
        G = RandomGinibre.matrices(n_states, [dim, dim])
        U = RandomUnitary().matrices(n_states, dims) + np.full((n_states, dim, dim), np.eye(dim))

        # state : (ID + U) G G^dag (ID + U^dag)
        G = U @ G
        G = G @ dagger(G)
        G /= np.trace(G, axis1=1, axis2=2)[:, None, None]

        return DMStack(G, dims), {}
