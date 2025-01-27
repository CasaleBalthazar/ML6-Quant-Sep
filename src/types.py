import numpy as np
import scipy.io as sio
from enum import IntEnum

class Label(IntEnum):
    """
    the standards labels for the separability problem
    """
    SEP = 0
    ENT = 1
    UNK = 2


class DMStack(np.ndarray):
    """
    a DMStack represent a stack of density matrices on a certain state space.
    It is a numpy.ndarray of shape (n_matrices, ...) and have n additional attribute dims which is a list indicating
    the dimension of the subsystem.

    The density matrices contained in a DMStack are not necessarily in the form of complex density matrices.
    """
    def __new__(cls, input_array, dims):
        obj = np.asarray(input_array).view(cls)
        obj.dims = dims

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.dims = getattr(obj, 'dims', None)


# save / load datasets

def save_dmstack(filename, stack, infos):
    df = {
        'dims' : stack.dims,
        'states' : stack,
        **infos
    }
    sio.savemat(filename, df)


def load_dmstack(filename):
    df = sio.loadmat(filename)
    dims = df.pop('dims')[0]
    states = df.pop('states')
    remove = ['__header__', '__version__', '__globals__']    # standard keys for sio
    for key in remove :
        df.pop(key)

    infos = {}
    for key in df.keys():
        infos[key] = np.squeeze(df[key])  # sio store 1D array as 2D matrices of the form [1,len(data)]

    return DMStack(states, dims), infos
