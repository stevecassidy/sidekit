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
import numpy as np
import copy
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from decimal import *
from scipy import ndimage
import logging
from sidekit.mixture import Mixture


__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2016 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def pre_emphasis(input_sig, pre):
    """Pre-emphasis of an audio signal.
    :param input_sig: the input vector of signal to pre emphasize
    :param pre: value that defines the pre-emphasis filter. 
    """
    return lfilter([1.0, -pre], 1, input_sig)


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    This method has been implemented by Anne Archibald, 
    as part of the talk box toolkit
    example::
    
        segment_axis(arange(10), 4, 2)
        array([[0, 1, 2, 3],
           ( [2, 3, 4, 5],
             [4, 5, 6, 7],
             [6, 7, 8, 9]])

    :param a: the array to segment
    :param length: the length of each frame
    :param overlap: the number of array elements by which the frames should overlap
    :param axis: the axis to operate on; if None, act on the flattened array
    :param end: what to do with the last frame, if the array is not evenly 
            divisible into pieces. Options are:
            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value

    :param endvalue: the value to use for end='pad'

    :return: a ndarray

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise ValueError("overlap must be nonnegative and length must" +
                         "be positive")

    if l < length or (l - length) % (length - overlap):
        if l > length:
            roundup = length + (1 + (l - length) // (length - overlap)) * (length - overlap)
            rounddown = length + ((l - length) // (length - overlap)) * (length - overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length - overlap) or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1, axis)

        if end == 'cut':
            a = a[..., :rounddown]
            l = a.shape[0]
        elif end in ['pad', 'wrap']:  # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup - l]
            a = b

        a = a.swapaxes(-1, axis)

    if l == 0:
        raise ValueError("Not enough data points to segment array " +
                         "in 'cut' mode; try 'pad' or 'wrap'")
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
    newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        logging.debug.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


def speech_enhancement(X, Gain, Noise_floor, Fs, Ascale, NN):
    """This program is only to process the single file seperated by the silence
    section if the silence section is detected, then a counter to number of
    buffer is set and pre-processing is required.

    Usage: SpeechENhance(wavefilename, Gain, Noise_floor)

    :param X: input audio signal
    :param Noise_floor: default value is 0.02 : suggestion range 
            from 0.2 to 0.001
    :param Gain: default value is 0.9, suggestion range 0.6 to 1.4,
            higher value means more subtraction or noise redcution
    :param Fs: sampling frequency of the input signal
    :param Ascale: 1 to add noise, 0 not to add noise
    :param NN:
    
    :return: a 1-dimensional array of boolean that 
        is True for high energy frames.
    
    Copyright 2014 Sun Han Wu and Anthony Larcher
    """
    # try{
    if X.shape[0] < 512:  # creer une exception
        return X

    num1 = 40  # dsiable buffer number
    Alpha = 0.75  # original value is 0.9
    FrameSize = 32 * 2  # 256*2
    FrameShift = int(FrameSize / NN)  # FrameSize/2=128
    nfft = FrameSize  # = FrameSize
    Fmax = int(np.floor(nfft / 2) + 1)  # 128+1 = 129
    # arising hamming windows
    Hamm = 1.08 * (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FrameSize) / (FrameSize - 1)))
    y0 = np.zeros(FrameSize - FrameShift)  # 128 zeros

    Eabsn = np.zeros(Fmax)
    Eta1 = Eabsn

    ###################################################################
    # initial parameter for noise min
    mb = np.ones((1 + FrameSize / 2, 4)) * FrameSize / 2  # 129x4  set four buffer * FrameSize/2
    im = 0
    Beta1 = 0.9024  # seems that small value is better;
    pxn = np.zeros(1 + FrameSize / 2)  # 1+FrameSize/2=129 zeros vector

    ###################################################################
    old_absx = Eabsn
    x = np.zeros(FrameSize)
    x[FrameSize - FrameShift:FrameSize] = X[
        np.arange(np.min((int(FrameShift), X.shape[0])))]  # fread(ifp, FrameSize, 'short')% read  FrameSize samples

    if x.shape[0] < FrameSize:
        EOF = 1
        return X

    EOF = 0
    Frame = 0

    ###################################################################
    # add the pre-noise estimates
    for i in range(200):
        Frame += 1
        fftn = fft(x * Hamm)  # get its spectrum
        absn = np.abs(fftn[0:Fmax])  # get its amplitude

        # add the following part from noise estimation algorithm
        pxn = Beta1 * pxn + (1 - Beta1) * absn  # Beta=0.9231 recursive pxn
        im = (im + 1) % 40  # noise_memory=47;  im=0 (init) for noise level estimation

        if im:
            mb[:, 0] = np.minimum(mb[:, 0], pxn)  # 129 by 4 im<>0  update the first vector from PXN
        else:
            mb[:, 1:] = mb[:, :3]  # im==0 every 47 time shift pxn to first vector of mb
            mb[:, 0] = pxn
            #  0-2  vector shifted to 1 to 3

        pn = 2 * np.min(mb, axis=1)  # pn = 129x1po(9)=1.5 noise level estimate compensation
        # over_sub_noise= oversubtraction factor

        # end of noise detection algotihm
        x[:FrameSize - FrameShift] = x[FrameShift:FrameSize]
        index1 = np.arange(FrameShift * Frame, np.min((FrameShift * (Frame + 1), X.shape[0])))
        In_data = X[index1]  # fread(ifp, FrameShift, 'short');

        if In_data.shape[0] < FrameShift:  # to check file is out
            EOF = 1
            break
        else:
            x[FrameSize - FrameShift:FrameSize] = In_data  # shift new 128 to position 129 to FrameSize location
            # end of for loop for noise estimation

    # end of prenoise estimation ************************
    x = np.zeros(FrameSize)
    x[FrameSize - FrameShift:FrameSize] = X[np.arange(np.min((int(FrameShift), X.shape[0])))]

    if x.shape[0] < FrameSize:
        EOF = 1
        return X

    EOF = 0
    Frame = 0

    X1 = np.zeros(X.shape)
    Frame = 0

    while EOF == 0:
        Frame += 1
        xwin = x * Hamm

        fftx = fft(xwin, nfft)  # FrameSize FFT
        absx = np.abs(fftx[0:Fmax])  # Fmax=129,get amplitude of x
        argx = fftx[:Fmax] / (absx + np.spacing(1))  # normalize x spectrum phase

        absn = absx

        # add the following part from rainer algorithm
        pxn = Beta1 * pxn + (1 - Beta1) * absn  # s Beta=0.9231   recursive pxn

        im = int((im + 1) % (num1 * NN / 2))  # original =40 noise_memory=47;  im=0 (init) for noise level estimation

        if im:
            mb[:, 0] = np.minimum(mb[:, 0], pxn)  # 129 by 4 im<>0  update the first vector from PXN
        else:
            mb[:, 1:] = mb[:, :3]  # im==0 every 47 time shift pxn to first vector of mb
            mb[:, 0] = pxn

        pn = 2 * np.min(mb, axis=1)  # pn = 129x1po(9)=1.5 noise level estimate compensation

        Eabsn = pn
        Gaina = Gain

        temp1 = Eabsn * Gaina

        Eta1 = Alpha * old_absx + (1 - Alpha) * np.maximum(absx - temp1, 0)
        new_absx = (absx * Eta1) / (Eta1 + temp1)  # wiener filter
        old_absx = new_absx

        ffty = new_absx * argx  # multiply amplitude with its normalized spectrum

        y = np.real(np.fft.fftpack.ifft(np.concatenate((ffty, np.conj(ffty[np.arange(Fmax - 2, 0, -1)])))))

        y[:FrameSize - FrameShift] = y[:FrameSize - FrameShift] + y0
        y0 = y[FrameShift:FrameSize]  # keep 129 to FrameSize point samples 
        x[:FrameSize - FrameShift] = x[FrameShift:FrameSize]

        index1 = np.arange(FrameShift * Frame, np.min((FrameShift * (Frame + 1), X.shape[0])))
        In_data = X[index1]  # fread(ifp, FrameShift, 'short');

        z = 2 / NN * y[:FrameShift]  # left channel is the original signal 
        z /= 1.15
        z = np.minimum(z, 32767)
        z = np.maximum(z, -32768)
        index0 = np.arange(FrameShift * (Frame - 1), FrameShift * Frame)
        if not all(index0 < X1.shape[0]):
            idx = 0
            while (index0[idx] < X1.shape[0]) & (idx < index0.shape[0]):
                X1[index0[idx]] = z[idx]
                idx += 1
        else:
            X1[index0] = z

        if In_data.shape[0] == 0:
            EOF = 1
        else:
            x[np.arange(FrameSize - FrameShift, FrameSize + In_data.shape[0] - FrameShift)] = In_data

    X1 = X1[X1.shape[0] - X.shape[0]:]
    # }
    # catch{

    # }
    return X1


def vad_energy(logEnergy,
               distribNb=3,
               nbTrainIt=8,
               flooring=0.0001, ceiling=1.0,
               alpha=2):
    # center and normalize the energy
    logEnergy = (logEnergy - np.mean(logEnergy)) / np.std(logEnergy)

    # Initialize a Mixture with 2 or 3 distributions
    world = Mixture()
    # set the covariance of each component to 1.0 and the mean to mu + meanIncrement
    world.cst = np.ones(distribNb) / (np.pi / 2.0)
    world.det = np.ones(distribNb)
    world.mu = -2 + 4.0 * np.arange(distribNb) / (distribNb - 1)
    world.mu = world.mu[:, np.newaxis]
    world.invcov = np.ones((distribNb, 1))
    # set equal weights for each component
    world.w = np.ones(distribNb) / distribNb
    world.cov_var_ctl = copy.deepcopy(world.invcov)

    # Initialize the accumulator
    accum = copy.deepcopy(world)

    # Perform nbTrainIt iterations of EM
    for it in range(nbTrainIt):
        accum._reset()
        # E-step
        world._expectation(accum, logEnergy)
        # M-step
        world._maximization(accum, ceiling, flooring)

    # Compute threshold
    threshold = world.mu.max() - alpha * np.sqrt(1.0 / world.invcov[world.mu.argmax(), 0])

    # Apply frame selection with the current threshold
    label = logEnergy > threshold
    return label


def vad_snr(sig, snr, fs=16000, shift=0.01, nwin=256):
    """Select high energy frames based on the Signal to Noise Ratio
    of the signal.
    
    :param sig: the input audio signal
    :param snr: Signal to noise ratio to consider
    :param fs: sampling frequency of the input signal in Hz. Default is 16000.
    :param shift: shift between two frames in seconds. Default is 0.01
    :param nwin: number of samples of the sliding window. Default is 256.
    """
    overlap = nwin - int(shift * fs)

    sig *= 32768

    sig = speech_enhancement(np.squeeze(sig), 1.2, 0.0, fs, 1.0, 2)
    # sig = wiener(sig, mysize=32)

    # Compute Standard deviation
    sig += 0.1 * np.random.randn(sig.shape[0])
    # std2 = sidekit.toFrame(sig / 32768, nwin, overlap).T
    # assume 16bit coding
    std2 = segment_axis(sig / 32768, nwin, overlap,
                        axis=None, end='cut', endvalue=0).T
    std2 = np.std(std2, axis=0)
    std2 = 20 * np.log10(std2)  # convert the dB

    # APPLY VAD
    label = (std2 > np.max(std2) - snr) & (std2 > -75)

    #####
    # Speech Enhancement    
    # sig = speech_enhancement(np.squeeze(sig), 1.2, 0.0, fs, 1.0, 2)

    # Compute Standard deviation
    # sig = sig + 0.001 * np.random.randn(sig.shape[0])
    # std2 = sidekit.toFrame(sig / 32768, nwin, overlap).T
    # assume 16bit coding
    # if norm:
    # std2 = segment_axis(sig / 32768, nwin, overlap,
    #                   axis=None, end='cut', endvalue=0).T
    # else:
    # std2 = segment_axis(sig, nwin, overlap,
    #                   axis=None, end='cut', endvalue=0).T
    # std2 = np.std(std2, axis=0)
    # std2 = 20 * np.log10(std2)  # convert the dB

    # APPLY VAD
    # label = (std2 > np.max(std2) - snr) & (std2 > -75)

    return label


def label_fusion(label, win=3):
    """Apply a morphological filtering on the label to remove isolated labels.
    In case the input is a two channel label (2D ndarray of boolean of same 
    length) the labels of two channels are fused to remove
    overlaping segments of speech.
    
    :param label: input labels given in a 1D or 2D ndarray
    :param win: parameter or the morphological filters
    """
    channel_nb = len(label)
    if channel_nb == 2:
        overlap_label = np.logical_and(label[0], label[1])
        label[0] = np.logical_and(label[0], ~overlap_label)
        label[1] = np.logical_and(label[1], ~overlap_label)

    for idx, lbl in enumerate(label):
        cl = ndimage.grey_closing(lbl, size=win)
        label[idx] = ndimage.grey_opening(cl, size=win)

    return label
