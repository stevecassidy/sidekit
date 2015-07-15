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
Copyright 2014-2015 Anthony Larcher and Sylvain Meignier

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""

__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2015 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'

import numpy as np
import scipy.stats as stats
from scipy.signal import lfilter


def rasta_filt(x):
    """Apply RASTA filtering to the input signal.
    
    :param x: the input audio signal to filter.
        cols of x = critical bands, rows of x = frame
        same for y but after filtering
        default filter is single pole at 0.94
    """
    x = x.T
    numer = np.arange(.2,-.3,-.1)
    denom = np.array([1,-0.98])

    # Initialize the state.  This avoids a big spike at the beginning
    # resulting from the dc offset level in each band.
    # (this is effectively what rasta/rasta_filt.c does).
    # Because Matlab uses a DF2Trans implementation, we have to
    # specify the FIR part to get the state right (but not the IIR part)
    y = np.zeros(x.shape)    
    zf = np.zeros((x.shape[0], 4))
    for i in range(y.shape[0]):
        #y[i, :4], zf[i, :4] = lfilter(numer, 1, x[i, :4], axis=-1, zi=[0, 0, 0, 0])
        y[i, :4], zf[i, :4] = lfilter(numer, 1, x[i, :4], axis=-1, zi=[0, 0, 0, 0])
    
    # .. but don't keep any of these values, just output zero at the beginning
    y = np.zeros(x.shape)

    # Apply the full filter to the rest of the signal, append it
    for i in range(y.shape[0]):
        y[i, 4:] = lfilter(numer, denom, x[i, 4:], axis=-1, zi=zf[i, :])[0]
    
    return y.T


def cms(features, label=[]):
    """Performs cepstral mean subtraction
    
    :param features: a feature stream of dimension dim x nframes 
            where dim is the dimension of the acoustic features and nframes the 
            number of frames in the stream
    :param label: a logical verctor

    :return: a feature stream
    """
    # If no label file as input: all speech are speech
    if label == []:
        label = np.ones(features.shape[0]).astype(bool)

    if all(label == False):
        normFeatures = features
    else:
        speechFeatures = features[label, :]
        mu = speechFeatures.mean(0)
    
        normFeatures = features - mu

    return normFeatures


def cmvn(features, label=[]):
    """Performs mean and variance normalization
    
    :param features: a feature stream of dimension dim x nframes 
        where dim is the dimension of the acoustic features and nframes the 
        number of frames in the stream
    :param label: a logical verctor

    :return: a sequence of features
    """
    # If no label file as input: all speech are speech
    if label == []:
        label = np.ones(features.shape[0]).astype(bool)

    if all(label == False):
        normFeatures = features
    else:
        speechFeatures = features[label, :]
        mu = speechFeatures.mean(0)
        stdev = np.std(speechFeatures, axis=0)
    
        normFeatures = features - mu
        normFeatures = normFeatures / stdev

    return normFeatures


def stg(features, label=[], win=301):
    """Performs feature warping on a sliding window
    
    :param features: a feature stream of dimension dim x nframes 
        where dim is the dimension of the acoustic features and nframes the
        number of frames in the stream

    :return: a sequence of features
    """

    # If no label file as input: all speech are speech
    if label == []:
        label = np.ones(features.shape[0]).astype(bool)
    speechFeatures = features[label, :]

    add_a_feature = False
    if win % 2 == 1:
        # one feature per line
        nframes, dim = np.shape(speechFeatures)

        # If the number of frames is not enough for one window
        if nframes < win:
            # if the number of frames is not odd, duplicate the last frame
            #if nframes % 2 == 1:
            if not nframes % 2 == 1:
                nframes += 1
                add_a_feature = True
                speechFeatures = np.concatenate((speechFeatures, [speechFeatures[-1, ]]))
            win = nframes

        # create the output feature stream
        stgFeatures = np.zeros(np.shape(speechFeatures))

        # Process first window
        R = np.argsort(speechFeatures[:win, ], axis=0)
        R = np.argsort(R, axis=0)
        arg = (R[: (win - 1) / 2] + 0.5) / win
        stgFeatures[: (win - 1) / 2, :] = stats.norm.ppf(arg, 0, 1)

        # process all follwing windows except the last one
        for m in range(int((win - 1) / 2), int(nframes - (win - 1) / 2)):
            idx = list(range(int(m - (win - 1) / 2),
                            int(m + (win - 1) / 2 + 1)))
            foo = speechFeatures[idx, :]
            R = np.sum(foo < foo[(win - 1) / 2], axis=0) + 1
            arg = (R - 0.5) / win
            stgFeatures[m, :] = stats.norm.ppf(arg, 0, 1)

        # Process the last window
        R = np.argsort(speechFeatures[list(range(nframes - win, nframes)), ], axis=0)
        R = np.argsort(R, axis=0)
        arg = (R[(win + 1) / 2: win, :] + 0.5) / win      
        
        stgFeatures[list(range(int(nframes - (win - 1) / 2), nframes)), ] \
                    = stats.norm.ppf(arg, 0, 1)
    else:
        # Raise an exception
        raise Exception('Sliding window should have an odd length')

    wrapFeatures = np.copy(features)
    if add_a_feature:
        stgFeatures = stgFeatures[:-1]
    wrapFeatures[label, :] = stgFeatures

    return wrapFeatures


def normalize_feature_stream(features, label=[], mode='cmvn', win='301',
                           normVar=False, keepAllFeatures=False):
    """Normalize features from a feature stream by using either 
    'cms', 'cmvn' or 'stg'

    :param features: a feature stream to normalize
    :param label: a logical vector True if the frame should be processed.
        By default, all frames are considered True.
    :param mode: normalization to apply: 'cms', 'cmvn' or 'stg'.
        Default is 'cmvn'.
    :param win: for 'stg' mode only, size of the sliding window.
        Default is 301.
    :param normVar: for 'cmvn' mode only, if True normalize the variance.
        Default is False.
    :param keepAllFeatures: boolean, if True, keep also non-processed features
    
    :return: a sequence of features
    """
    # if no label, use all features
    if label == []:
        label = np.ones(features.shape[0], dtype='bool')

    if mode == 'cmvn':
        features = cmvn(features, label=label)
    elif mode == 'cms':
        features = cms(features, label=label)
    elif mode == 'stg':
        speechFeatures = features[label, :]
        speechFeatures = stg(speechFeatures, win)
        features[label, :] = speechFeatures

    if keepAllFeatures:
        normfeatures = features
    else:
        normfeatures = features[label, :]

    return normfeatures


