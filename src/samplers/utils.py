"""
Contain utility functions for sampling
"""

from ..types import DMStack

import numpy as np


def kron(m1, m2):
    """
    element-wise kronecker product
    :param m1: stack of matrices 1
    :param m2: stack of matrices 2
    :return: kronecker product element wise
    """
    return np.array([np.kron(m1[i], m2[i]) for i in range(len(m1))])


def dagger(m):
    """
    element-wise conjugate transpose of a set of matrices
    :param m: stack of complex-valued matrices
    :return: their conjugate transpose element-wise
    """
    return np.transpose(m, axes=(0,2,1)).conjugate()


def balanced(n_states, n_params):
    """
    number of states to be sampled per parameter
    :param n_states: number of states to be sampled in total
    :param n_params: number of parameters
    :return: an array of int
    """
    n_per_val = np.full(n_params, int(n_states/n_params))
    remaining = n_states - np.sum(n_per_val)
    while remaining > 0 :
        remaining -= 1
        n_per_val[np.random.randint(0, n_params)] += 1
    return n_per_val


class RandomGinibre :
    """
    random matrices where each element is a complex number following the normal distribution
    """
    @staticmethod
    def matrices(n_mats, dims):
        """
        :param n_mats: number of matrices
        :param dims: dimensions of the matrices
        :return: a set of random complex-valued matrices
        """
        return np.random.randn(n_mats, *dims) + 1j * np.random.randn(n_mats, *dims)


class RandomUnitary :
    """
    random unitary matrices from the Haar measure in a certain state space
    """
    def __init__(self, product=False):
        """

        :param product: if True, the matrices will be of the form A \otimes B
        """
        self.product = product

    def matrices(self, n_mats, dims):
        """
        sample a set of random Unitary matrices
        :param n_mats: number of matrices
        :param dims: dimensions of the state space
        :return: a array [n_mats, dim, dim] of unitary matrices
        """
        if self.product :
            s1 = RandomUnitary().matrices(n_mats, [dims[0]])
            s2 = RandomUnitary().matrices(n_mats, [dims[1]])
            return kron(s1,s2)
        else :
            dim = np.product(dims)
            return np.linalg.qr(RandomGinibre.matrices(n_mats, [dim, dim]), mode='complete')[0]


class FromSet :
    """
    Sample random elements from a preexisting set
    """
    def __init__(self, states, infos):
        """
        :param states: states in the set (DMStack)
        :param infos: dictionary of information
        """
        self.datas = states
        self.infos = infos

    def states(self, n_states, dims=None):
        """
        sample a random set of n_states states from a preexisting set
        :param n_states: number of states to be sampled
        :param dims: dimensions of the state space (unused)
        :return: a set of states (DMStack) and their relevant informations (dict)
        """
        dims = self.datas.dims

        if n_states >= len(self.datas) :
            return DMStack(self.datas.copy(), dims), self.infos.copy()

        idx = np.arange(len(self.datas))
        np.random.shuffle(idx)
        idx = idx[:n_states]

        states = self.datas.copy()[idx]
        infos = self.infos.copy()
        for key in infos.keys() :
            infos[key] = infos[key][idx]

        return DMStack(states, dims), infos
