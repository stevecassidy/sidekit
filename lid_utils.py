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
Copyright 2014-2015 Anthony Larcher

:mod:`lid_utils` provides utilities to perform Language identification.
"""

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2015 Anthony Larcher"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'

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


def Gaussian_Backend_Train(train_ss):
    """
    Take a StatServer of training examples as input
    output a StatServer mean for each class and a tied co-variance matrix
    """
    
    #Compute parameters of the Gaussian backend (common covariance and constant)
    vectSize = train_ss.stat1.shape[1]
    uniqueSpeaker = np.unique(train_ss.modelset)
    gb_sigma = train_ss.get_within_covariance_stat1()
    
    # Compute mean of each class
    gb_mean = train_ss.mean_stat_per_model()
    
    # Compute the normalization constant
    gb_cst = - 0.5 * (np.linalg.slogdet(gb_sigma)[1] \
                      + train_ss.stat1.shape[1] * np.log(2*np.pi))
    
    return gb_mean, gb_sigma, gb_cst


def Gaussian_Backend_Test(test_ss, params, diag=False):
    """
    
    For a diaonal Back-End only but input covariance can be full or diagonal
    """

    gb_mean, gb_sigma, gb_cst = params
    
    scores = Scores()
    scores.modelset = gb_mean.modelset
    scores.segset = test_ss.segset
    scores.scoremat = np.ones((gb_mean.modelset.shape[0], test_ss.segset.shape[0]))
    scores.scoremask = np.ones(scores.scoremat.shape, dtype='bool')

    if diag:
        gb_gmm = sidekit.Mixture()
        gb_gmm.w = np.ones(gb_mean.modelset.shape[0], dtype='float') /  gb_mean.modelset.shape[0]
        gb_gmm.mu = gb_mean.stat1
        if gb_sigma.ndim == 2:
            gb_gmm.invcov = np.tile(1 / np.diag(gb_sigma), (gb_mean.modelset.shape[0], 1))
        elif gb_sigma.ndim == 2:
            gb_gmm.invcov = np.tile(1 / gb_sigma, (gb_mean.modelset.shape[0], 1))
        gb_gmm._compute_all()
        
        scores.scoremat = gb_gmm.compute_log_posterior_probabilities(test_ss.stat1)

    else:
        assert gb_sigma.ndim == 2
        scores.scoremat *= gb_cst

        inv_sigma = np.linalg.inv(gb_sigma)

        # Compute scores for all trials per language
        for lang in range(gb_mean.modelset.shape[0]):
            scores.scoremat[lang , :] -= 0.5 * (gb_mean.stat1[lang, :].dot(inv_sigma).dot(gb_mean.stat1[lang, :].T)
                                    -2 * np.sum(test_ss.stat1.dot(inv_sigma) * gb_mean.stat1[lang, :], axis=1)
                                    + np.sum(test_ss.stat1.dot(inv_sigma) * test_ss.stat1, axis=1))
        
    assert scores.validate()
    return scores








