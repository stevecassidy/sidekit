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

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""
import copy
import logging
import numpy
from scipy.fftpack import fft
from scipy import ndimage
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
    if input_sig.ndim == 1:
        return (input_sig - numpy.c_[input_sig[numpy.newaxis, :][..., :1],
                                     input_sig[numpy.newaxis, :][..., :-1]].squeeze() * pre)
    else:
        return input_sig - numpy.c_[input_sig[..., :1], input_sig[..., :-1]] * pre


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
        a = numpy.ravel(a)  # may copy
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
            b = numpy.empty(s, dtype=a.dtype)
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
    new_shape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
    new_strides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]

    try:
        return numpy.ndarray.__new__(numpy.ndarray, strides=new_strides,
                                     shape=new_shape, buffer=a, dtype=a.dtype)
    except TypeError:
        logging.debug("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        new_strides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]
        return numpy.ndarray.__new__(numpy.ndarray, strides=new_strides,
                                     shape=new_shape, buffer=a, dtype=a.dtype)


def speech_enhancement(x, gain, nn):
    """This program is only to process the single file separated by the silence
    section if the silence section is detected, then a counter to number of
    buffer is set and pre-processing is required.

    :param x: input audio signal
    :param gain: default value is 0.9, suggestion range 0.6 to 1.4,
            higher value means more subtraction or noise reduction
    :param nn:
    
    :return: a 1-dimensional array of boolean that 
        is True for high energy frames.
    
    Copyright 2014 Sun Han Wu and Anthony Larcher
    """
    if x.shape[0] < 512:
        return x

    num1 = 40  # disable buffer number
    alpha = 0.75  # original value is 0.9
    frame_size = 32 * 2  # 256*2
    frame_shift = int(frame_size / nn)  # frame_size/2=128
    n_fft = frame_size  # = frame_size
    f_max = int(numpy.floor(n_fft / 2) + 1)  # 128+1 = 129
    # arising hamming windows
    hamm = 1.08 * (0.54 - 0.46 * numpy.cos(2 * numpy.pi * numpy.arange(frame_size) / (frame_size - 1)))
    y0 = numpy.zeros(frame_size - frame_shift)  # 128 zeros

    eabsn = numpy.zeros(f_max)

    ###################################################################
    # initial parameter for noise min
    mb = numpy.ones((1 + int(frame_size / 2), 4)) * frame_size / 2  # 129x4  set four buffer * frame_size/2
    im = 0
    beta1 = 0.9024  # seems that small value is better;
    pxn = numpy.zeros(1 + int(frame_size / 2))  # 1+frame_size/2=129 zeros vector

    ###################################################################
    old_absx = eabsn
    x = numpy.zeros(frame_size)
    # fread(ifp, frame_size, 'short')% read  frame_size samples
    x[frame_size - frame_shift:frame_size] = x[numpy.arange(numpy.min((int(frame_shift), x.shape[0])))]

    if x.shape[0] < frame_size:
        return x

    frame = 0

    ###################################################################
    # add the pre-noise estimates
    for i in range(200):
        frame += 1
        fftn = fft(x * hamm)  # get its spectrum
        absn = numpy.abs(fftn[0:f_max])  # get its amplitude

        # add the following part from noise estimation algorithm
        pxn = beta1 * pxn + (1 - beta1) * absn  # Beta=0.9231 recursive pxn
        im = (im + 1) % 40  # noise_memory=47;  im=0 (init) for noise level estimation

        if im:
            mb[:, 0] = numpy.minimum(mb[:, 0], pxn)  # 129 by 4 im<>0  update the first vector from PXN
        else:
            mb[:, 1:] = mb[:, :3]  # im==0 every 47 time shift pxn to first vector of mb
            mb[:, 0] = pxn
            #  0-2  vector shifted to 1 to 3

        # pn = 2 * numpy.min(mb, axis=1)  # pn = 129x1po(9)=1.5 noise level estimate compensation
        # over_sub_noise= oversubtraction factor

        # end of noise detection algotihm
        x[:frame_size - frame_shift] = x[frame_shift:frame_size]
        index1 = numpy.arange(frame_shift * frame, numpy.min((frame_shift * (frame + 1), x.shape[0])))
        in_data = x[index1]  # fread(ifp, frame_shift, 'short');

        if in_data.shape[0] < frame_shift:  # to check file is out
            break
        else:
            x[frame_size - frame_shift:frame_size] = in_data  # shift new 128 to position 129 to frame_size location
            # end of for loop for noise estimation

    # end of prenoise estimation ************************
    x = numpy.zeros(frame_size)
    x[frame_size - frame_shift:frame_size] = x[numpy.arange(numpy.min((int(frame_shift), x.shape[0])))]

    if x.shape[0] < frame_size:
        return x

    eof = 0
    x1 = numpy.zeros(x.shape)
    frame = 0

    while eof == 0:
        frame += 1
        xwin = x * hamm

        fftx = fft(xwin, n_fft)  # frame_size FFT
        absx = numpy.abs(fftx[0:f_max])  # f_max=129,get amplitude of x
        argx = fftx[:f_max] / (absx + numpy.spacing(1))  # normalize x spectrum phase

        absn = absx

        # add the following part from rainer algorithm
        pxn = beta1 * pxn + (1 - beta1) * absn  # s Beta=0.9231   recursive pxn

        im = int((im + 1) % (num1 * nn / 2))  # original =40 noise_memory=47;  im=0 (init) for noise level estimation

        if im:
            mb[:, 0] = numpy.minimum(mb[:, 0], pxn)  # 129 by 4 im<>0  update the first vector from PXN
        else:
            mb[:, 1:] = mb[:, :3]  # im==0 every 47 time shift pxn to first vector of mb
            mb[:, 0] = pxn

        pn = 2 * numpy.min(mb, axis=1)  # pn = 129x1po(9)=1.5 noise level estimate compensation

        eabsn = pn
        gaina = gain

        temp1 = eabsn * gaina

        eta1 = alpha * old_absx + (1 - alpha) * numpy.maximum(absx - temp1, 0)
        new_absx = (absx * eta1) / (eta1 + temp1)  # wiener filter
        old_absx = new_absx

        ffty = new_absx * argx  # multiply amplitude with its normalized spectrum

        y = numpy.real(numpy.fft.fftpack.ifft(numpy.concatenate((ffty, numpy.conj(ffty[numpy.arange(f_max - 2, 0, -1)])))))

        y[:frame_size - frame_shift] = y[:frame_size - frame_shift] + y0
        y0 = y[frame_shift:frame_size]  # keep 129 to frame_size point samples
        x[:frame_size - frame_shift] = x[frame_shift:frame_size]

        index1 = numpy.arange(frame_shift * frame, numpy.min((frame_shift * (frame + 1), x.shape[0])))
        in_data = x[index1]  # fread(ifp, frame_shift, 'short');

        z = 2 / nn * y[:frame_shift]  # left channel is the original signal
        z /= 1.15
        z = numpy.minimum(z, 32767)
        z = numpy.maximum(z, -32768)
        index0 = numpy.arange(frame_shift * (frame - 1), frame_shift * frame)
        if not all(index0 < x1.shape[0]):
            idx = 0
            while (index0[idx] < x1.shape[0]) & (idx < index0.shape[0]):
                x1[index0[idx]] = z[idx]
                idx += 1
        else:
            x1[index0] = z

        if in_data.shape[0] == 0:
            eof = 1
        else:
            x[numpy.arange(frame_size - frame_shift, frame_size + in_data.shape[0] - frame_shift)] = in_data

    x1 = x1[x1.shape[0] - x.shape[0]:]
    return x1


def vad_percentil(log_energy, percent):
    """

    :param log_energy:
    :param percent:
    :return:
    """
    thr = numpy.percentile(log_energy, percent)
    return log_energy > thr, thr


def vad_energy(log_energy,
               distrib_nb=3,
               nb_train_it=8,
               flooring=0.0001, ceiling=1.0,
               alpha=2):
    # center and normalize the energy
    log_energy = (log_energy - numpy.mean(log_energy)) / numpy.std(log_energy)

    # Initialize a Mixture with 2 or 3 distributions
    world = Mixture()
    # set the covariance of each component to 1.0 and the mean to mu + meanIncrement
    world.cst = numpy.ones(distrib_nb) / (numpy.pi / 2.0)
    world.det = numpy.ones(distrib_nb)
    world.mu = -2 + 4.0 * numpy.arange(distrib_nb) / (distrib_nb - 1)
    world.mu = world.mu[:, numpy.newaxis]
    world.invcov = numpy.ones((distrib_nb, 1))
    # set equal weights for each component
    world.w = numpy.ones(distrib_nb) / distrib_nb
    world.cov_var_ctl = copy.deepcopy(world.invcov)

    # Initialize the accumulator
    accum = copy.deepcopy(world)

    # Perform nbTrainIt iterations of EM
    for it in range(nb_train_it):
        accum._reset()
        # E-step
        world._expectation(accum, log_energy)
        # M-step
        world._maximization(accum, ceiling, flooring)

    # Compute threshold
    threshold = world.mu.max() - alpha * numpy.sqrt(1.0 / world.invcov[world.mu.argmax(), 0])

    # Apply frame selection with the current threshold
    label = log_energy > threshold
    return label, threshold


def vad_snr(sig, snr, fs=16000, shift=0.01, nwin=256):
    """Select high energy frames based on the Signal to Noise Ratio
    of the signal.
    Input signal is expected encoded on 16 bits
    
    :param sig: the input audio signal
    :param snr: Signal to noise ratio to consider
    :param fs: sampling frequency of the input signal in Hz. Default is 16000.
    :param shift: shift between two frames in seconds. Default is 0.01
    :param nwin: number of samples of the sliding window. Default is 256.
    """
    overlap = nwin - int(shift * fs)

    sig /= 32768.

    sig = speech_enhancement(numpy.squeeze(sig), 1.2, 2)

    # Compute Standard deviation
    sig += 0.1 * numpy.random.randn(sig.shape[0])
    std2 = segment_axis(sig , nwin, overlap, axis=None, end='cut', endvalue=0).T
    std2 = numpy.std(std2, axis=0)
    std2 = 20 * numpy.log10(std2)  # convert the dB

    # APPLY VAD
    label = (std2 > numpy.max(std2) - snr) & (std2 > -75)

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
        overlap_label = numpy.logical_and(label[0], label[1])
        label[0] = numpy.logical_and(label[0], ~overlap_label)
        label[1] = numpy.logical_and(label[1], ~overlap_label)

    for idx, lbl in enumerate(label):
        cl = ndimage.grey_closing(lbl, size=win)
        label[idx] = ndimage.grey_opening(cl, size=win)

    return label
