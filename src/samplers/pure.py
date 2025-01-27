"""
Contain method to sample pure quantum states
"""

import numpy as np

from .utils import RandomGinibre, kron, dagger

class RandomHaar :
    """
    Random density matrices sampled from the Haar measure
    """
    def __init__(self, product=False):
        """
        :param product: if True, the matrices will be of the form A \otimes B
                        where A, B sampled from the Haar measure
        """
        self.product = product

    def states(self, n_states, dims):
        """
        sample states from the Haar measure
        :param n_states: number of states
        :param dims: dimensions of the subsystems
        :return: a set of random states (DMStack), a information dictionary (empty)
        """
        if self.product :
            s1, _ = RandomHaar().states(n_states, [dims[0]])
            s2, _ = RandomHaar().states(n_states, [dims[1]])
            return kron(s1, s2), {}
        else :
            dim = np.product(dims)
            states = RandomGinibre.matrices(n_states, [dim, 1])
            states = states @ dagger(states)
            states /= np.trace(states, axis1=1, axis2=2)[:,None,None]

            return states, {}
