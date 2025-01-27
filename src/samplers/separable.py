"""
Contain methods to sample separable density matrices
"""
from ..types import DMStack
from .utils import kron, balanced

import numpy as np

class RandomSeparable:
    """
    Randomly generated density matrices separable by construction
    """

    def __init__(self, nsums, sampl_mthds):
        """
        :param nsums: int of array of int (number of member states)
        :param sampl_mthds: list of 2 samplers from which the states will be sampled
        """
        self.nsums = nsums
        self.sampl_mthds = sampl_mthds

    def states(self, n_states, dims):
        """
        sample n_states random separable density matrices
        :param n_states: number of states
        :param dims: dimensions of the subsystems
        :return: a set of density matrices (DMStack) and a dictionary of information (n_sum : key 'nsum')
        """
        if isinstance(self.nsums, int):
            return self.states_nsum(self.nsums, n_states, dims)
        else :
            return self.states_list(self.nsums, n_states, dims)

    def states_nsum(self, nsum, n_states, dims):
        dim = np.product(dims)

        probas = np.random.rand(n_states, nsum)
        probas /= np.sum(probas, axis=1)[:,None]

        states = np.zeros((n_states, dim, dim), dtype=complex)
        for i in range(nsum):
            s1,_ = self.sampl_mthds[0](n_states, [dims[0]])
            s2,_ = self.sampl_mthds[1](n_states, [dims[1]])
            states += probas[:,i,None,None] * kron(s1,s2)
        return DMStack(states,dims), {'nsum' : np.full(n_states, nsum)}

    def states_list(self, vals_nsum, n_states, dims):
        dim = np.product(dims)
        per_n = balanced(n_states, len(vals_nsum))

        states = np.zeros((n_states, dim, dim), dtype=complex)
        nsums = np.zeros(n_states, dtype=int)
        idx = 0
        for i in range(len(per_n)):
            if per_n[i] == 0:
                continue
            states[idx : idx + per_n[i]] = self.states_nsum(vals_nsum[i], per_n[i], dims)[0]
            nsums[idx : idx + per_n[i]] = vals_nsum[i]
            idx += per_n[i]
        return DMStack(states, dims), {'nsum' : nsums}
