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

:mod:`svm_scoring` provides functions to perform speaker verification 
by using Support Vector Machines.
"""
import os
import sys
import numpy as np
import threading
import logging
import sidekit.sv_utils
from sidekit.bosaris import Ndx
from sidekit.bosaris import Scores
from sidekit.statserver import StatServer

if sys.version_info.major == 3:
    import queue as Queue
else:
    import Queue

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def svm_scoring_singleThread(svmDir, test_sv, ndx, score, segIdx=[]):
    """Compute scores for SVM verification on a single thread
    (two classes only as implementeed at the moment)
     
    :param svmDir: directory where to load the SVM models
    :param test_sv: StatServer object of super-vectors. stat0 are set to 1 and stat1 are the super-vector to classify
    :param ndx: Ndx object of the trials to perform
    :param score: Scores object to fill
    :param segIdx: list of segments to classify. Classify all if the list is empty.
    """ 
    assert os.path.isdir(svmDir), 'First parameter should be a directory'
    assert isinstance(test_sv, StatServer), 'Second parameter should be a StatServer'
    assert isinstance(ndx, Ndx), 'Third parameter should be an Ndx'

    if not segIdx:
        segIdx = range(ndx.segset.shape[0])

    # Load SVM models
    Msvm = np.zeros((ndx.modelset.shape[0], test_sv.stat1.shape[1]))
    bsvm = np.zeros(ndx.modelset.shape[0])
    for m in range(ndx.modelset.shape[0]):
        svmFileName = os.path.join(svmDir, ndx.modelset[m] + '.svm')
        w, b = sidekit.sv_utils.read_svm(svmFileName)
        Msvm[m, :] = w
        bsvm[m] = b

    # Compute scores against all test segments
    for ts in segIdx:
        logging.info('Compute trials involving test segment %d/%d', ts + 1, ndx.segset.shape[0])

        # Select the models to test with the current segment
        models = ndx.modelset[ndx.trialmask[:, ts]]
        ind_dict = dict((k, i) for i, k in enumerate(ndx.modelset))
        inter = set(ind_dict.keys()).intersection(models)
        idx_ndx = np.array([ind_dict[x] for x in inter])

        scores = np.dot(Msvm[idx_ndx, :], test_sv.stat1[ts, :]) + bsvm[idx_ndx]

        # Fill the score matrix
        score.scoremat[idx_ndx, ts] = scores


def svm_scoring(svmDir, test_sv, ndx, numThread=1):
    """Compute scores for SVM verification on multiple threads
    (two classes only as implementeed at the moment)
    
    :param svmDir: directory where to load the SVM models
    :param test_sv: StatServer object of super-vectors. stat0 are set to 1 and stat1
          are the super-vector to classify
    :param ndx: Ndx object of the trials to perform
    :param numThread: number of thread to launch in parallel
    
    :return: a Score object.
    """
    # Remove missing models and test segments
    existingModels, modelIdx = sidekit.sv_utils.check_file_list(ndx.modelset, svmDir, '.svm')
    clean_ndx = ndx.filter(existingModels, test_sv.segset, True)

    Score = Scores()
    Score.scoremat = np.zeros(clean_ndx.trialmask.shape)
    Score.modelset = clean_ndx.modelset
    Score.segset = clean_ndx.segset
    Score.scoremask = clean_ndx.trialmask

    # Split the list of segment to process for multi-threading
    los = np.array_split(np.arange(clean_ndx.segset.shape[0]), numThread)

    jobs = []
    for idx in los:
        p = threading.Thread(target=svm_scoring_singleThread, 
                             args=(svmDir, test_sv, ndx, Score, idx))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    return Score
