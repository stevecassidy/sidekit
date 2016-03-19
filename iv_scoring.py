# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.
"""
Copyright 2014-2016 Anthony Larcher and Sylvain Meignier

    :mod:`iv_scoring` provides methods to compare i-vectors
"""

import numpy as np
import scipy as sp
import copy
from sidekit.bosaris import Ndx
from sidekit.bosaris import Scores
from sidekit.statserver import StatServer
import logging

import sys
if sys.version_info.major > 2 :
    from functools import reduce


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def cosine_scoring(enroll, test, ndx, wccn=None):
    """Compute the cosine similarities between to sets of vectors. The list of 
    trials to perform is given in an Ndx object.
    
    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param wccn: numpy.ndarray, if provided, the i-vectors are normalized by using a Within Class Covariance Matrix
    
    :return: a score object
    """
    assert isinstance(enroll, StatServer), 'First parameter should be a StatServer'
    assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    enroll_copy = copy.deepcopy(enroll)
    test_copy = copy.deepcopy(test)

    # Remove missing models and test segments
    clean_ndx = ndx.filter(enroll_copy.modelset, test_copy.segset, True)

    # Align StatServers to match the clean_ndx
    enroll_copy.align_models(clean_ndx.modelset)
    test_copy.align_segments(clean_ndx.segset)

    if wccn is not None:
        enroll_copy.rotate_stat1(wccn)
        if enroll_copy != test_copy:
            test_copy.rotate_stat1(wccn)

    # Cosine scoring
    enroll_copy.norm_stat1()
    if enroll_copy != test_copy:
        test_copy.norm_stat1()
    S = np.dot(enroll_copy.stat1, test_copy.stat1.transpose())

    Score = Scores()
    Score.scoremat = S
    Score.modelset = clean_ndx.modelset
    Score.segset = clean_ndx.segset
    Score.scoremask = clean_ndx.trialmask
    return Score


def mahalanobis_scoring(enroll, test, ndx, M):
    """Compute the mahalanobis distance between to sets of vectors. The list of 
    trials to perform is given in an Ndx object.
    
    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param M: mahalanobis matrix as a ndarray
    
    :return: a score object
    """
    assert isinstance(enroll, StatServer), 'First parameter should be a StatServer'
    assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    assert enroll.stat1.shape[1] == test.stat1.shape[1], 'I-vectors dimension mismatch'
    assert enroll.stat1.shape[1] == M.shape[0], 'I-vectors and Mahalanobis matrix dimension mismatch'
    # Remove missing models and test segments
    clean_ndx = ndx.filter(enroll.modelset, test.segset, True)

    # Align StatServers to match the clean_ndx
    enroll.align_models(clean_ndx.modelset)
    test.align_segments(clean_ndx.segset)

    # Mahalanobis scoring
    S = np.zeros((enroll.modelset.shape[0], test.segset.shape[0]))
    for i in range(enroll.modelset.shape[0]):
        diff = enroll.stat1[i, :] - test.stat1
        S[i, :] = -0.5 * np.sum(np.dot(diff, M) * diff, axis=1)

    score = Scores()
    score.scoremat = S
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask
    return score


def two_covariance_scoring(enroll, test, ndx, W, B):
    """Compute the 2-covariance scores between to sets of vectors. The list of 
    trials to perform is given in an Ndx object. Within and between class 
    co-variance matrices have to be pre-computed.
    
    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param W: the within-class co-variance matrix to consider
    :param B: the between-class co-variance matrix to consider
      
    :return: a score object
    """
    assert isinstance(enroll, StatServer), 'First parameter should be a directory'
    assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    assert enroll.stat1.shape[1] == test.stat1.shape[1], 'I-vectors dimension mismatch'
    assert enroll.stat1.shape[1] == W.shape[0], 'I-vectors and co-variance matrix dimension mismatch'
    assert enroll.stat1.shape[1] == B.shape[0], 'I-vectors and co-variance matrix dimension mismatch'

    # Remove missing models and test segments
    clean_ndx = ndx.filter(enroll.modelset, test.segset, True)

    # Align StatServers to match the clean_ndx
    enroll.align_models(clean_ndx.modelset)
    test.align_segments(clean_ndx.segset)

    # Two covariance scoring scoring
    S = np.zeros((enroll.modelset.shape[0], test.segset.shape[0]))
    iW = sp.linalg.inv(W)
    iB = sp.linalg.inv(B)

    G = reduce(np.dot, [iW, sp.linalg.inv(iB + 2*iW), iW])
    H = reduce(np.dot, [iW, sp.linalg.inv(iB + iW), iW])

    s2 = np.sum(np.dot(enroll.stat1, H) * enroll.stat1, axis=1)
    s3 = np.sum(np.dot(test.stat1, H) * test.stat1, axis=1)

    for ii in range(enroll.modelset.shape[0]):
        A = enroll.stat1[ii, :] + test.stat1
        s1 = np.sum(np.dot(A, G) * A, axis=1)
        S[ii, :] = s1 - s3 - s2[ii]

    score = Scores()
    score.scoremat = S
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask
    return score


def PLDA_scoring(enroll, test, ndx, mu, F, G, Sigma, P_known=0.0):
    """Compute the PLDA scores between to sets of vectors. The list of 
    trials to perform is given in an Ndx object. PLDA matrices have to be 
    pre-computed. i-vectors are supposed to be whitened before.
    
    Implements the appraoch described in [Lee13]_ including scoring 
    for partially open-set identification
    
    :param enroll: a StatServer in which stat1 are i-vectors
    :param test: a StatServer in which stat1 are i-vectors
    :param ndx: an Ndx object defining the list of trials to perform
    :param mu: the mean vector of the PLDA gaussian
    :param F: the between-class co-variance matrix of the PLDA
    :param G: the within-class co-variance matrix of the PLDA
    :param Sigma: the residual covariance matrix
    :param P_known: probability of having a known speaker for open-set
        identification case (=1 for the verification task and =0 for the 
        closed-set case)
      
    :return: a score object
    """
    assert isinstance(enroll, StatServer), 'First parameter should be a StatServer'
    assert isinstance(test, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'
    assert enroll.stat1.shape[1] == test.stat1.shape[1], 'I-vectors dimension mismatch'
    assert enroll.stat1.shape[1] == F.shape[0], 'I-vectors and co-variance matrix dimension mismatch'
    assert enroll.stat1.shape[1] == G.shape[0], 'I-vectors and co-variance matrix dimension mismatch'
    
    enroll_copy = copy.deepcopy(enroll)
    test_copy = copy.deepcopy(test)
    
    # Remove missing models and test segments
    clean_ndx = ndx.filter(enroll_copy.modelset, test_copy.segset, True)
    
    # Align StatServers to match the clean_ndx
    enroll_copy.align_models(clean_ndx.modelset)
    test_copy.align_segments(clean_ndx.segset)

    # Center the i-vectors around the PLDA mean
    enroll_copy.center_stat1(mu)
    test_copy.center_stat1(mu)
        
    # If models are not unique, compute the mean per model, display a warning
    if not np.unique(enroll_copy.modelset).shape == enroll_copy.modelset.shape:
        logging.warning("Enrollment models are not unique, average i-vectors")
        enroll_copy = enroll_copy.mean_stat_per_model()
    
    # Compute temporary matrices
    invSigma = np.linalg.inv(Sigma)
    I_iv = np.eye(mu.shape[0], dtype='float')
    I_ch = np.eye(G.shape[1], dtype='float')
    I_spk = np.eye(F.shape[1], dtype='float')
    A = np.linalg.inv(G.T.dot(invSigma).dot(G) + I_ch)
    B = F.T.dot(invSigma).dot(I_iv - G.dot(A).dot(G.T).dot(invSigma))
    K = B.dot(F)
    K1 = np.linalg.inv(K + I_spk)
    K2 = np.linalg.inv(2 * K + I_spk)
    
    # Compute the Gaussian distribution constant
    alpha1 = np.linalg.slogdet(K1)[1]
    alpha2 = np.linalg.slogdet(K2)[1]
    constant = alpha2 / 2.0 - alpha1
    
    # Compute verification scores
    score = Scores()
    score.scoremat = np.zeros(clean_ndx.trialmask.shape)
    score.modelset = clean_ndx.modelset
    score.segset = clean_ndx.segset
    score.scoremask = clean_ndx.trialmask
    
    # Project data in the space that maximizes the speaker separability
    test_tmp = B.dot(test_copy.stat1.T)
    enroll_tmp = B.dot(enroll_copy.stat1.T)

    # Compute verification scores
    # Loop on the models
    for model_idx in range(enroll_copy.modelset.shape[0]):
    
        s2 = enroll_tmp[:, model_idx].dot(K1).dot(enroll_tmp[:, model_idx])
        
        mod_plus_test_seg = test_tmp + np.atleast_2d(enroll_tmp[:, model_idx]).T
    
        tmp1 = test_tmp.T.dot(K1)
        tmp2 = mod_plus_test_seg.T.dot(K2)
        
        for seg_idx in range(test_copy.segset.shape[0]):
            s1 = tmp1[seg_idx, :].dot(test_tmp[:, seg_idx])
            s3 = tmp2[seg_idx, :].dot(mod_plus_test_seg[:, seg_idx])
            score.scoremat[model_idx, seg_idx] = (s3 - s1 - s2)/2. + constant

    # Case of open-set identification, we compute the log-likelihood 
    # by taking into account the probability of having a known impostor
    # or an out-of set class
    if P_known != 0:
        N = score.scoremat.shape[0]
        open_set_scores = np.empty(score.scoremat.shape)
        tmp = np.exp(score.scoremat)
        for ii in range(N):
            open_set_scores[ii, :] = score.scoremat[ii, :] \
                - np.log(P_known * tmp[~(np.arange(N) == ii)].sum(axis=0) / (N - 1) + (1 - P_known))  # open-set term
        score.scoremat = open_set_scores

    return score





