"""
Contain approximate methods to detect separability
"""

from ..types import Label

import numpy as np


class MlModel :
    """
    Use a machine learning model (sklearn) as model
    """
    def __init__(self, model) :
        self.model = model

    def predict(self, states, infos={}):
        return self.model.predict(state), {}

class DistToSep:
    """
    Distance from a separable approximation
    """

    def __init__(self, dist_threshold, sep_key=None, sep_mthd=None):
        """
        :param dist_threshold: minimum distance to be considered entangled
        :param sep_key: key containing the separable approximation in the info dictionary
        :param sep_mthd: method to compute the separable approximation (if key_sep == None)
        """
        self.dist_threshold = dist_threshold
        self.sep_key = sep_key
        self.sep_mthd = sep_mthd

    def predict(self, states, infos={}):
        """
        predict a state as separable or entangled in function of its distance to a separable approximation
        :param states: quantum density matrices
        :param infos: dictionary of information
        :return: array of Label
        """
        aprx = None
        inf_aprx = {}
        return_aprx = False
        if self.sep_key is not None :
            aprx = infos[self.sep_key]
        elif self.sep_mthd is not None :
            return_aprx = True
            aprx, inf_aprx = self.sep_mthd(states, infos)

        y = np.full(len(states), Label.ENT)
        y[np.linalg.norm(aprx - states, axis=(1,2)) < self.dist_threshold] = Label.SEP

        if return_aprx :
            return y, {'aprx' : aprx, **inf_aprx}
        else :
            return y, {}


class WitQuality:
    """
    Quality of an approximate entanglement witness
    """
    def __init__(self, min_score, sep_test_set, wit_key=None, wit_mthd=None, return_scores = False):
        """

        :param min_score: minimum score for the witness to be considered valid
        :param sep_test_set: DMStack of separable states on which the witness will be tested
        :param wit_key: key containing the witness in the information dictionary
        :param wit_mthd: method to compute the witness (if wit_key == None)
        :param return_scores: return or not the score achieved by each witness as info (key : 'wit_score')
        """
        self.min_score = min_score
        self.test_set = sep_test_set
        self.wit_key = wit_key
        self.wit_mthd = wit_mthd
        self.return_scores = return_scores

    def predict(self, states, infos={}):
        """
        predict a state as separable or entangled in function of the quality of an approximate entanglement witness
        :param states: quantum density matrices
        :param infos: dictionary of informations
        :return: array of Label
        """
        wits = None
        inf_wits = {}
        return_wit = False
        if self.wit_key is not None :
            wits = infos[self.wit_key]
        elif self.wit_mthd is not None :
            wits, inf_wits = self.wit_mthd(states, infos)

        resp = np.full(len(states), True)
        # tr(W_rho rho) < 0
        resp = np.logical_and(resp, np.trace(wits @ states, axis1=1, axis2=2).real < 0)

        # tr(W_rho sigma) >= 0 for vast majority of separables sigma
        scores = np.zeros(len(states))
        for i in range(len(wits)) :
            scores[i] = np.average(np.trace(np.matmul(wits[i], self.test_set), axis1=1, axis2=2).real >= 0)
        resp = np.logical_and(resp, scores > self.min_score)

        if self.return_scores :
            inf_wits = {'wit_score' : scores, **inf_wits}

        if return_wit :
            inf_wits = {'wit' : wits, **inf_wits}

        y = np.full(len(states), Label.SEP)
        y[resp] = Label.ENT

        return y, inf_wits
