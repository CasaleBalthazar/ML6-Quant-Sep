"""
Contain various separability criteria / entanglement witnesses
"""

from ..types import Label

import numpy as np

class PPT:
    """
    The positive partial transpose criterion
    """

    @staticmethod
    def is_respected(states, infos={}):
        """
        indicate if the states respect the PPT criterion
        :param states: quantum density matrices
        :param infos: information dictionary (unused)
        :return: array of boolean
        """
        tol = 1e-10 # avoid errors due to precision
        eigvals = np.linalg.eigvalsh(PPT.partial_transpose(states, infos))
        return np.all(eigvals > -tol, axis=1), {}

    @staticmethod
    def predict(states, infos={}):
        """
        predict the state as entangled, separable or unknown using the PPT criterion
        :param states: quantum density matrices
        :param infos: information dictionary (unused)
        :return: array of Label
        """
        y = np.full(len(states), Label.ENT)
        if np.product(states.dims) <= 6 :
            y[PPT.is_respected(states, infos)[0]] = Label.SEP
        else :
            y[PPT.is_respected(states, infos)[0]] = Label.UNK
        return y, {}

    @staticmethod
    def partial_transpose(states, infos={}):
        dims = states.dims
        return np.transpose(states.reshape(states.shape[0], dims[0], dims[1], dims[0], dims[1]),
                            axes=(0, 1, 4, 3, 2)).reshape(states.shape)


# TODO : implement
"""
class Realignment:
    @staticmethod
    def is_respected(states, infos={}):
        s = np.linalg.svd(Realignment.realigned(states, infos), compute_uv=False)
        return np.sum(s, axis=1) <= 1, {}

    @staticmethod
    def predict(states, infos={}):
        y = np.full(len(states), Label.ENT)
        y[Realignment.is_respected(states, infos)[0]] = Label.UNK
        return y, {}

    @staticmethod
    def realigned(states, infos={}):
        dims = states.dims
        blocks = states.reshape(len(states), dims[0], dims[1], dims[0], dims[1])
        result = np.zeros((len(states), dims[0]**2, dims[1]**2), dtype=complex)

        idx = 0
        for i in range(dims[0]**2) :
            for j in range(dims[1]**2) :
                result[:, ] = blocks[:, i, j].reshape((len(states), )
                idx += 1
        return result
"""

class SepBall:
    """
    The ball of separability around the maximally mixed state
    """
    @staticmethod
    def contain(states, infos={}):
        """
        indicate if a state is inside the ball of separability around the maximally mixed state
        :param states: quantum density matrices
        :param infos: information dictionary (unused)
        :return: array of boolean
        """
        dim = np.product(states.dims)
        purity = np.trace(states @ states, axis1=1, axis2=2).real
        return purity <= 1 / (dim - 1), {}

    @staticmethod
    def predict(states, infos={}):
        """
        predict the state as separable, entangled or unknown using the separable ball criterion
        :param states: quantum density matrices
        :param infos: information dictionary (unused)
        :return: array of Label
        """
        y = np.full(len(states), Label.UNK)
        y[SepBall.contain(states, infos)[0]] = Label.SEP
        return y, {}


class EntWitnesses :
    """
    A set of entanglement witnesses
    """
    def __init__(self, witnesses):
        """
        :param witnesses: a set of valid entanglement witnesses on a state space
        """
        self.witnesses = witnesses

    def witness(self, states, infos={}):
        """
        indicate if a state is detected as entangled by at least one witness in the set
        :param states: quantum density matrices
        :param infos: information dictionary (unused)
        :return: array of boolean
        """
        y = np.full(len(states), False)
        for i in range(len(self.witnesses)) :
            traces = np.trace(np.matmul(self.witnesses[i], states), axis1=1, axis2=2).real
            y = np.logical_or(y, traces < 0)
        return y

    def predict(self, states, infos={}):
        y = np.full(len(states), Label.UNK)
        y[self.witness(states, infos)[0]] = Label.ENT
        return y
