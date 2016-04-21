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
Copyright 2014-2016 Anthony Larcher

:mod:`lid_utils` provides utilities to perform Language identification.
"""

import numpy as np
import scipy as sp
import pickle
import gzip
import os

from sidekit.mixture import Mixture
from sidekit.statserver import StatServer
from sidekit.features_server import FeaturesServer
from sidekit.bosaris import Ndx
from sidekit.bosaris import Scores
import sidekit.sv_utils
import sidekit.frontend

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def log_sum_exp(x):
    """
    :param x: input vector
    """
    m, n = x.shape
    xmax = x.max(axis=0)
    xnorm = x - xmax
    ex = np.exp(xnorm)
    return xmax + np.log(ex.sum(axis=0))


def compute_log_likelihood_ratio(M, P_tar=0.5):
    """
    Compute log-likelihood ratio for closed-set identification.
    
    :param M: a matrix of log-likelihood of shape nb_models x nb_test_segments
    :param P_tar: probability of a trial to be from a target
    
    :return: a matrix of log-likelihood ration of shape 
        nb_models x nb_test_segments
    """
    llr = np.empty(M.shape)
    log_prior = np.ones((M.shape[0] - 1, 1)) * np.log((1 - P_tar) / (M.shape[0] - 1))
    for ii in range(M.shape[0]):
        llr[ii, :] = np.log(P_tar) + M[ii, :] - log_sum_exp(M[~(np.arange(M.shape[0]) == ii)] + log_prior)

    return llr


def Gaussian_Backend_Train(train_ss):
    """
    Take a StatServer of training examples as input
    output a StatServer mean for each class and a full tied co-variance matrix
    :param train_ss: sidekit.StatServer containing training data
    """

    # Compute parameters of the Gaussian backend (common covariance and constant)
    vectSize = train_ss.stat1.shape[1]
    uniqueSpeaker = np.unique(train_ss.modelset)
    gb_sigma = train_ss.get_within_covariance_stat1()

    # Compute mean of each class
    gb_mean = train_ss.mean_stat_per_model()

    # Compute the normalization constant
    gb_cst = - 0.5 * (np.linalg.slogdet(gb_sigma)[1] + train_ss.stat1.shape[1] * np.log(2 * np.pi))

    return gb_mean, gb_sigma, gb_cst


def Gaussian_Backend_Train_Hetero(train_ss, alpha=0.1):
    """
    Take a StatServer of training examples as input
    output a StatServer mean for each class and a full tied co-variance matrix
    :param train_ss: sidekit.StatServer of input training data
    :param alpha: weight of the a priori distribution learned on all training data
    """

    # Compute parameters of the Gaussian backend (common covariance and constant)
    vectSize = train_ss.stat1.shape[1]
    uniqueLanguage = np.unique(train_ss.modelset)
    # gb_sigma = train_ss.get_within_covariance_stat1()

    W = np.zeros((vectSize, vectSize))
    gb_sigma = []

    for languageID in uniqueLanguage:
        spkCtrVec = train_ss.get_model_stat1(languageID) \
                    - np.mean(train_ss.get_model_stat1(languageID), axis=0)
        gb_sigma.append(np.dot(spkCtrVec.transpose(), spkCtrVec))
        W += gb_sigma[-1]
        gb_sigma[-1] /= spkCtrVec.shape[0]
    W /= train_ss.stat1.shape[0]

    for ii in range(len(gb_sigma)):
        gb_sigma[ii] = alpha * gb_sigma[ii] + (1 - alpha) * W

    # Compute mean of each class
    gb_mean = train_ss.mean_stat_per_model()

    # Compute the normalization constant
    gb_cst = []
    for ii in range(len(gb_sigma)):
        gb_cst.append(- 0.5 * (np.linalg.slogdet(gb_sigma[ii])[1] + train_ss.stat1.shape[1] * np.log(2 * np.pi)))

    return gb_mean, gb_sigma, gb_cst


def _Gaussian_Backend_Train(data, label):
    """
    Take a StatServer of training examples as input
    output a StatServer mean for each class and a tied co-variance matrix
    """
    train_ss = StatServer()
    train_ss.segset = label
    train_ss.modelset = label
    train_ss.stat1 = data
    train_ss.stat0 = np.ones((data.shape[0], 1))
    train_ss.start = np.empty(data.shape[0], dtype="object")
    train_ss.stop = np.empty(data.shape[0], dtype="object")

    return Gaussian_Backend_Train(train_ss)


def Gaussian_Backend_Test(test_ss, params, diag=False, compute_llr=True):
    """
    Process data through a Gaussian-Backend which parameters (mean and variance)
    have been estimated using Gaussian_Backend_Train.
    
    If compute_llr is set to true, return the log-likelihood ratio, if not,
    return rthe log-likelihood on each Gaussian distrbution. Default is True.
    
    :param test_ss: a StatServer which stat1 are vectors to classify
    :param params: Gaussian Backend parameters, a tupple of mean, covariance
        and constante computed with Gaussian_Backend_Train
    :param diag: boolean, if true: use the diagonal version of the covariance
        matrix, if not the full version
    :param compute_llr: boolean, if true, return the log-likelihood ratio, if not,
    return rthe log-likelihood on each Gaussian distrbution.
    """
    gb_mean, gb_sigma, gb_cst = params

    scores = Scores()
    scores.modelset = gb_mean.modelset
    scores.segset = test_ss.segset
    scores.scoremat = np.ones((gb_mean.modelset.shape[0], test_ss.segset.shape[0]))
    scores.scoremask = np.ones(scores.scoremat.shape, dtype='bool')

    if diag:
        gb_gmm = Mixture()
        gb_gmm.w = np.ones(gb_mean.modelset.shape[0], dtype='float') / gb_mean.modelset.shape[0]
        gb_gmm.mu = gb_mean.stat1
        if gb_sigma.ndim == 2:
            gb_gmm.invcov = np.tile(1 / np.diag(gb_sigma), (gb_mean.modelset.shape[0], 1))
        elif gb_sigma.ndim == 2:
            gb_gmm.invcov = np.tile(1 / gb_sigma, (gb_mean.modelset.shape[0], 1))
        gb_gmm._compute_all()

        scores.scoremat = gb_gmm.compute_log_posterior_probabilities(test_ss.stat1).T

    else:
        assert gb_sigma.ndim == 2
        scores.scoremat *= gb_cst

        inv_sigma = np.linalg.inv(gb_sigma)

        # Compute scores for all trials per language
        for lang in range(gb_mean.modelset.shape[0]):
            scores.scoremat[lang, :] -= 0.5 * (gb_mean.stat1[lang, :].dot(inv_sigma).dot(gb_mean.stat1[lang, :].T) -
                                               2 * np.sum(test_ss.stat1.dot(inv_sigma) * gb_mean.stat1[lang, :],
                                                          axis=1) +
                                               np.sum(test_ss.stat1.dot(inv_sigma) * test_ss.stat1, axis=1))

    if compute_llr:
        scores.scoremat = compute_log_likelihood_ratio(scores.scoremat)

    assert scores.validate()
    return scores


def Gaussian_Backend_Test_Hetero(test_ss, params, diag=False, compute_llr=True):
    """
    Process data through a Gaussian-Backend which parameters (mean and variance)
    have been estimated using Gaussian_Backend_Train.
    
    If compute_llr is set to true, return the log-likelihood ratio, if not,
    return rthe log-likelihood on each Gaussian distrbution. Default is True.
    
    :param test_ss: a StatServer which stat1 are vectors to classify
    :param params: Gaussian Backend parameters, a tupple of mean, covariance
        and constante computed with Gaussian_Backend_Train
    :param diag: boolean, if true: use the diagonal version of the covariance
        matrix, if not the full version
    :param compute_llr: boolean, if true, return the log-likelihood ratio, if not,
    return rthe log-likelihood on each Gaussian distrbution.
    """
    gb_mean, gb_sigma, gb_cst = params

    scores = sidekit.Scores()
    scores.modelset = gb_mean.modelset
    scores.segset = test_ss.segset
    scores.scoremat = np.ones((gb_mean.modelset.shape[0], test_ss.segset.shape[0]))
    scores.scoremask = np.ones(scores.scoremat.shape, dtype='bool')

    if diag:

        gb_gmm = sidekit.Mixture()
        gb_gmm.w = np.ones(gb_mean.modelset.shape[0], dtype='float') / gb_mean.modelset.shape[0]
        gb_gmm.mu = gb_mean.stat1
        gb_gmm.invcov = np.empty(gb_gmm.mu.shape)
        for l in range(len(gb_sigma)):
            if gb_sigma[0].ndim == 2:
                gb_gmm.invcov[l, :] = 1 / np.diag(gb_sigma[l])
            elif gb_sigma[0].ndim == 1:
                gb_gmm.invcov[l, :] = 1 / gb_sigma[l]
        gb_gmm._compute_all()

        scores.scoremat = gb_gmm.compute_log_posterior_probabilities(test_ss.stat1).T

    else:
        assert gb_sigma[0].ndim == 2
        for lang in range(gb_mean.modelset.shape[0]):
            scores.scoremat[lang, :] *= gb_cst[lang]

        inv_sigma = np.linalg.inv(gb_sigma)

        # Compute scores for all trials per language
        for lang in range(gb_mean.modelset.shape[0]):
            scores.scoremat[lang, :] -= 0.5 * (gb_mean.stat1[lang, :].dot(inv_sigma[lang]).dot(gb_mean.stat1[lang, :].T)
                                               - 2 * np.sum(test_ss.stat1.dot(inv_sigma[lang]) * gb_mean.stat1[lang, :],
                                                            axis=1) + \
                                               np.sum(test_ss.stat1.dot(inv_sigma[lang]) * test_ss.stat1, axis=1))

    if compute_llr:
        scores.scoremat = compute_log_likelihood_ratio(scores.scoremat)

    assert scores.validate()
    return scores
