from types import DMStack

import numpy as np
import time

class Pipeline:
    """
    A Pipeline is a sampler which apply a serie of transformations to an initial sample.
    A Pipeline repeatedly sample from the same distribution until the desired number of sample is obtained.
    If the Pipeline produce more samples than initially desired, then it will return the n_sample first samples.
    """
    def __init__(self, steps, name = '', verbosity = 0, max_batch_size = None, sep_name='__'):
        """
        :param steps: a list of tuple (name, transfo_func). The first member of the list must be a tuple (name, sample_func).
        :param name: name of the pipeline (printed if verbose)
        :param verbosity: number of message outputed :
            0 - no message
            1 - a message at every batch (nb of states to be sampled / nb of states produced  and time taken)
            2 - a message at every step (nb of states produced and time taken)
        :param max_batch_size: if not None, limit the number of states produced each batch.
        :param sep name: any new information produced by a step will be added to the dictionary info in the key
                         step_name + sep_name + key (equal to '__' by default)
        """
        self.name = name
        self.steps = steps
        self.max_batch_size = max_batch_size
        self.verbose = verbosity
        self.sep_name = sep_name

    def sample(self, n_samples, dims=None):
        states = None
        infos = None

        n_sampled = 0

        if self.verbose >= 1:
            print(f'BEGIN GENERATION PROCESS {self.name} [{n_samples}]')


        # BATCHES CREATION
        while n_sampled < n_samples:
            batch_size = self.batch_size(n_samples, n_sampled)

            if self.verbose >= 1:
                print(f'BEGIN BATCH [{batch_size}]...', end='')
                if self.verbose >= 2:
                    print('')

            batch_beg = time.time()

            name, sample = self.steps[0]

            if self.verbose >= 2:
                print(f'{name} step...', end='')

            beg = time.time()
            batch_states, batch_infos = sample(batch_size, dims)
            end = time.time()

            if self.verbose >= 2:
                print(f'[{len(batch_states)}] (time : {end - beg:.2f} s)')

            # formate initial infos keys
            keys = list(batch_infos.keys())
            for key in keys:
                batch_infos[name + self.sep_name + key] = batch_infos.pop(key)

            add = True
            # perform each steps
            for i in range(1, len(self.steps)):

                name, transfo = self.steps[i]

                if self.verbose >= 2:
                    print(f'{name} step...', end='')

                beg = time.time()
                batch_states, batch_infos, new_infos = transfo(batch_states, batch_infos)
                end = time.time()

                if self.verbose >= 2:
                    print(f'done. [{len(batch_states)}] (time : {end - beg:.2f} s)')

                # no states remaining at the end of the step : make another batch.
                if len(batch_states) == 0:
                    add = False
                    break

                for key in new_infos.keys():
                    batch_infos[name + self.sep_name + key] = new_infos[key]

            batch_end = time.time()
            if self.verbose >= 1:
                print(f'END BATCH [{len(batch_states)}] ({batch_end - batch_beg:.2f} s)')

            # add states to previous samples.
            if add:
                if states is None:
                    states = batch_states
                    infos = batch_infos
                else:
                    states = DMStack(np.concatenate((states, batch_states)), dims)
                    for key in infos.keys():
                        infos[key] = np.concatenate((infos[key], batch_infos[key]))
                n_sampled = len(states)

        if n_sampled > n_samples:
            states = DMStack(states[:n_samples], dims)
            for key in infos.keys():
                infos[key] = infos[key][:n_samples]

        if self.verbose >= 1:
            print(f'END GENERATION PROCESS {self.name} [{len(states)}]')


        return states, infos

    def batch_size(self, n_samples, n_sampled):
        if self.max_batch_size is None:
            return n_samples - n_sampled
        else:
            return min(n_samples - n_sampled, self.max_batch_size)


# utility functions

# select the states associated to the desired label by a classification function
def select(label_func, value):
    """
    select the states associated to a certain label by a model function
    :param label_func: a model function
    :param value: desired label
    :return: tuple (states, infos, new_infos)
    """
    def transform(states, infos={}) :
        labels, infos_labels = label_func(states,infos)
        states = states[labels == value]

        for key in infos.keys():
            infos[key] = infos[key][labels == value]
        for key in infos_labels.keys():
            infos_labels[key] = infos_labels[key][labels == value]
        return states, infos, infos_labels

    return transform

# add the transformation of a state as additional informations
def add(transfo_func, key):
    """
    add the result of a transformer function as a key in the information dictionary
    :param transfo_func: a transformer function
    :param key: the key in which the result will be added
    :return: tuple (state, infos, new_infos)
    """
    def transform(states, infos={}):
        tran, tran_infos = transfo_func(states, infos)
        return states, infos, {key : tran, **tran_infos}

    return transform

# replace the states by their transformation
def apply(transfo_func):
    """
    replace the states of the set by the result of a transformer function
    :param transfo_func: a transformer function
    :return: tuple (state, infos, new_infos)
    """
    def transform(states, infos={}):
        tran, tran_infos = transfo_func(states, infos)
        return tran, infos, tran_infos
    return transform

# order the state by increasing ('+') or decreasing ('-') value of an information.
def order(key, first='+'):
    """
    order the states of the set by increasing/decreasing order of the content in a key
    :param key: the key containing the values
    :param first: '+' : increasing order, '-' : decreasing order
    :return: a tuple (states, infos, new_infos)
    """
    def transform(states, infos={}):
        indexes = np.argsort(infos[key])
        if first == '-':
            indexes = np.flip(indexes)
        states = states[indexes]
        for ikey in infos.keys():
            infos[ikey] = infos[ikey][indexes]
        return states, infos, {}

    return transform

# randomly shuffle the states
def shuffle():
    """
    randomly shuffle the states of the set
    :return: a tuple (states, infos, new_infos)
    """
    def transform(states, infos={}):
        indexes = np.arange(len(states))
        np.random.shuffle(indexes)
        states = states[indexes]
        for key in infos.keys():
            infos[key] = infos[key][indexes]
        return states, infos, {}

    return transform
