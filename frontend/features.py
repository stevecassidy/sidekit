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

import numpy as np
import scipy
from scipy.signal import hamming
from scipy.fftpack.realtransforms import dct
from sidekit.frontend.vad import *
from sidekit.frontend.io import *
from sidekit.frontend.normfeat import *
from sidekit.frontend.features import *

# from memory_profiler import profile
import gc

__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2016 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def hz2mel(f):
    """Convert an array of frequency in Hz into mel.
    
    :param f: frequency to convert
    
    :return: the equivalene on the mel scale.
    """
    return 1127.01048 * np.log(f / 700.0 + 1)


def mel2hz(m):
    """Convert an array of mel values in Hz.
    
    :param m: ndarray of frequencies to convert in Hz.
    
    :return: the equivalent values in Hertz.
    """
    return (np.exp(m / 1127.01048) - 1) * 700.0


def compute_delta(features, win=3, method='filter',
                  filt=np.array([.25, .5, .25, 0, -.25, -.5, -.25])):
    """features is a 2D-ndarray  each row of features is a a frame
    
    :param features: the feature frames to compute the delta coefficients
    :param win: parameter that set the length of the computation window.
            The eize of the window is (win x 2) + 1
    :param method: method used to compute the delta coefficients
        can be diff or filter
    :param filt: definition of the filter to use in "filter" mode, default one
        is similar to SPRO4:  filt=np.array([.2, .1, 0, -.1, -.2])
        
    :return: the delta coefficients computed on the original features.
    """
    # First and last features are appended to the begining and the end of the 
    # stream to avoid border effect
    x = np.zeros((features.shape[0] + 2 * win, features.shape[1]))
    x[:win, :] = features[0, :]
    x[win:-win, :] = features
    x[-win:, :] = features[-1, :]

    delta = np.zeros(x.shape)

    if method == 'diff':
        filt = np.zeros(2 * win + 1)
        filt[0] = -1
        filt[-1] = 1

    for i in range(features.shape[1]):
        delta[:, i] = np.convolve(features[:, i], filt)

    return delta[win:-win, :]


def pca_dct(cep, left_ctx=12, right_ctx=12, P=None):
    """Apply DCT PCA as in [McLaren 2015] paper:
    Mitchell McLaren and Yun Lei, 'Improved Speaker Recognition 
    Using DCT coefficients as features' in ICASSP, 2015
    
    A 1D-dct is applied to the cepstral coefficients on a temporal
    sliding window.
    The resulting matrix is then flatten and reduced by using a Principal
    Component Analysis.
    
    :param cep: a matrix of cepstral cefficients, 1 line per feature vector
    :param left_ctx: number of frames to consider for left context
    :param right_ctx: number of frames to consider for right context
    :param P: a PCA matrix trained on a developpment set to reduce the 
       dimension of the features. P is a portait matrix
    """
    y = np.r_[np.resize(cep[0, :], (left_ctx, cep.shape[1])),
              cep,
              np.resize(cep[-1, :], (right_ctx, cep.shape[1]))]

    ceps = framing(y, win_size=left_ctx + 1 + right_ctx).transpose(0, 2, 1)
    dct_temp = (dct_basis(left_ctx + 1 + right_ctx, left_ctx + 1 + right_ctx)).T
    if P is None:
        P = np.eye(dct_temp.shape[0] * cep.shape[1])
    return (np.dot(ceps.reshape(-1, dct_temp.shape[0]),
                   dct_temp).reshape(ceps.shape[0], -1)).dot(P)


def shifted_delta_cepstral(cep, d=1, P=3, k=7):
    """
    Compute the Shifted-Delta-Cepstral features for language identification
    
    :param cep: matrix of feature, 1 vector per line
    :param d: represents the time advance and delay for the delta computation
    :param k: number of delta-cepstral blocks whose delta-cepstral 
       coefficients are stacked to form the final feature vector
    :param P: time shift between consecutive blocks.
    
    return: cepstral coefficient concatenated with shifted deltas
    """

    y = np.r_[np.resize(cep[0, :], (d, cep.shape[1])),
              cep,
              np.resize(cep[-1, :], (k * 3 + d, cep.shape[1]))]

    delta = compute_delta(y, win=d, method='diff')

    sdc = np.empty((cep.shape[0], cep.shape[1] * k))

    idx = np.zeros(len(sdc), dtype='bool')
    for ii in range(k):
        idx[d + ii * P] = True
    for ff in range(len(cep)):
        sdc[ff, :] = delta[idx, :].reshape(1, -1)
        idx = np.roll(idx, 1)
    return np.hstack((cep, sdc))


def trfbank(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt, midfreq=1000):
    """Compute triangular filterbank for cepstral coefficient computation.

    :param fs: sampling frequency of the original signal.
    :param nfft: number of points for the Fourier Transform
    :param lowfreq: lower limit of the frequency band filtered
    :param maxfreq: higher limit of the frequency band filtered
    :param nlinfilt: number of linear filters to use in low frequencies
    :param  nlogfilt: number of log-linear filters to use in high frequencies
    :param midfreq: frequency boundary between linear and log-linear filters

    :return: the filter bank and the central frequencies of each filter
    """
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    # ------------------------
    # Compute the filter bank
    # ------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt + 2)
    if nlogfilt == 0:
        linsc = (maxfreq - lowfreq) / (nlinfilt + 1)
        freqs[:nlinfilt + 2] = lowfreq + np.arange(nlinfilt + 2) * linsc
    elif nlinfilt == 0:
        lowMel = hz2mel(lowfreq)
        maxMel = hz2mel(maxfreq)
        mels = np.zeros(nlogfilt + 2)
        mels[nlinfilt:]
        melsc = (maxMel - lowMel) / (nfilt + 1)
        mels[:nlogfilt + 2] = lowMel + np.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        freqs = mel2hz(mels)
    else:
        # Compute linear filters on [0;1000Hz]
        linsc = (min([midfreq, maxfreq]) - lowfreq) / (nlinfilt + 1)
        freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
        # Compute log-linear filters on [1000;maxfreq]
        lowMel = hz2mel(min([1000, maxfreq]))
        maxMel = hz2mel(maxfreq)
        mels = np.zeros(nlogfilt + 2)
        melsc = (maxMel - lowMel) / (nlogfilt + 1)

        # Verify that mel2hz(melsc)>linsc
        while mel2hz(melsc) < linsc:
            # in this case, we add a linear filter
            nlinfilt += 1
            nlogfilt -= 1
            freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
            lowMel = hz2mel(freqs[nlinfilt - 1] + 2 * linsc)
            maxMel = hz2mel(maxfreq)
            mels = np.zeros(nlogfilt + 2)
            melsc = (maxMel - lowMel) / (nlogfilt + 1)

        mels[:nlogfilt + 2] = lowMel + np.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        freqs[nlinfilt:] = mel2hz(mels)

    heights = 2. / (freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, np.floor(nfft / 2) + 1))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i + 1]
        hi = freqs[i + 2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        min(np.floor(hi * nfft / fs) + 1, nfft), dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid[:-1]] = rslope * (hi - nfreqs[rid[:-1]])

    return fbank, freqs



def mel_filter_bank(fs, nfft, lowfreq, maxfreq, widest_nlogfilt, widest_lowfreq, widest_maxfreq,):
    """Compute triangular filterbank for cepstral coefficient computation.

    :param fs: sampling frequency of the original signal.
    :param nfft: number of points for the Fourier Transform
    :param lowfreq: lower limit of the frequency band filtered
    :param maxfreq: higher limit of the frequency band filtered
    :param nlinfilt: number of linear filters to use in low frequencies
    :param  nlogfilt: number of log-linear filters to use in high frequencies

    :return: the filter bank and the central frequencies of each filter
    """

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    widest_freqs = np.zeros(widest_nlogfilt + 2)

    lowMel = hz2mel(widest_lowfreq)
    maxMel = hz2mel(widest_maxfreq)
    mels = np.zeros(widest_nlogfilt+2)
    melsc = (maxMel - lowMel)/ (widest_nlogfilt + 1)
    mels[:widest_nlogfilt + 2] = lowMel + np.arange(widest_nlogfilt + 2) * melsc
    # Back to the frequency domain
    widest_freqs = mel2hz(mels)

    # Select filters in the narrow band
    sub_band_freqs = np.array([fr for fr in widest_freqs if lowfreq <= fr <= maxfreq])

    heights = 2./(sub_band_freqs[2:] - sub_band_freqs[0:-2])
    nfilt = sub_band_freqs.shape[0] - 2

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, np.floor(nfft/2)+1))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = sub_band_freqs[i]
        cen = sub_band_freqs[i+1]
        hi = sub_band_freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        min(np.floor(hi * nfft / fs) + 1,nfft), dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid[:-1]] = rslope * (hi - nfreqs[rid[:-1]])

    return fbank, sub_band_freqs


def mfcc(input_sig, lowfreq=100, maxfreq=8000, nlinfilt=0, nlogfilt=24,
         nwin=0.025, fs=16000, nceps=13, shift=0.01,
         get_spec=False, get_mspec=False):
    """Compute Mel Frequency Cepstral Coefficients.

    :param input_sig: input signal from which the coefficients are computed.
            Input audio is supposed to be RAW PCM 16bits
    :param lowfreq: lower limit of the frequency band filtered. 
            Default is 100Hz.
    :param maxfreq: higher limit of the frequency band filtered.
            Default is 8000Hz.
    :param nlinfilt: number of linear filters to use in low frequencies.
            Default is 0.
    :param nlogfilt: number of log-linear filters to use in high frequencies.
            Default is 24.
    :param nwin: length of the sliding window in seconds
            Default is 0.025.
    :param fs: sampling frequency of the original signal. Default is 16000Hz.
    :param nceps: number of cepstral coefficients to extract. 
            Default is 13.
    :param shift: shift between two analyses. Default is 0.01 (10ms).
    :param get_spec: boolean, if true returns the spectrogram
    :param get_mspec:  boolean, if true returns the output of the filter banks
    :return: the cepstral coefficients in a ndaray as well as 
            the Log-spectrum in the mel-domain in a ndarray.

    .. note:: MFCC are computed as follows:
        
            - Pre-processing in time-domain (pre-emphasizing)
            - Compute the spectrum amplitude by windowing with a Hamming window
            - Filter the signal in the spectral domain with a triangular filter-bank, whose filters are approximatively
               linearly spaced on the mel scale, and have equal bandwith in the mel scale
            - Compute the DCT of the log-spectrom
            - Log-energy is returned as first coefficient of the feature vector.
    
    For more details, refer to [Davis80]_.
    """

    # Pre-emphasis factor (to take into account the -6dB/octave rolloff of the
    # radiation at the lips level
    prefac = 0.
    extract = pre_emphasis(input_sig, prefac)

    # Compute the overlap of frames and cut the signal in frames of length nwin
    # overlaping by "overlap" samples
    window_length = int(round(nwin * fs))
    #window_length = int((nwin * fs))
    w = hamming(window_length, sym=0)
    overlap = window_length - int(shift * fs)
    framed = segment_axis(extract, window_length, overlap)

    l = framed.shape[0]
    nfft = 2 ** int(np.ceil(np.log2(window_length)))

    spec = np.ones((l, nfft / 2 + 1))
    logEnergy = np.ones(l)

    dec = 10000
    start = 0
    stop = min(dec, l)
    while start < l:
        # Compute the spectrum magnitude
        tmp = framed[start:stop, :] * w
        spec[start:stop, :] = np.abs(np.fft.rfft(tmp, nfft, axis=-1))

        # Compute the log-energy of each frame
        logEnergy[start:stop] = 2.0 * np.log(np.sqrt(np.sum(np.square(tmp), axis=1)))
        start = stop
        stop = min(stop + dec, l)

    del framed
    del extract

    # Filter the spectrum through the triangle filterbank
    # Prepare the hamming window and the filter bank
    fbank = trfbank(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt)[0]
    # mspec = np.log(np.maximum(1.0, np.dot(spec, fbank.T)))
    mspec = np.log10(np.dot(spec, fbank.T))
    del fbank

    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    # The C0 term is removed as it is the constant term
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, 1:nceps + 1]
    lst = list()
    lst.append(ceps)
    lst.append(logEnergy)
    if get_spec:
        lst.append(spec)
    else:
        lst.append(None)
        del spec
    if get_mspec:
        lst.append(mspec)
    else:
        lst.append(None)
        del mspec

    return lst


def framing(sig, win_size, win_shift=1, context=(0,0), pad='zeros'):
    """
    :param sig: input signal, can be mono or multi dimensional
    :param win_size: size of the window in term of samples
    :param context: tuple of left and right context
    :param pad: can be zeros or edge
    """
    dsize = sig.dtype.itemsize
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]
    # Manage padding
    c = (context,) +  (sig.ndim - 1) * ((0,0),)
    _win_size = win_size + sum(context)
    shape = ((sig.shape[0] - win_size) / win_shift + 1, 1, _win_size, sig.shape[1])
    strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    return np.lib.stride_tricks.as_strided(np.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                                    shape=shape,
                                                    strides=strides).squeeze()

def dct_basis(nbasis, length):
    """
    :param nbasis: number of CT coefficients to keep
    :param length: length of the matrix to process
    :return: a basis of DCT coefficients
    """
    return scipy.fftpack.idct(np.eye(nbasis, length), norm='ortho')


def get_trap(X, left_ctx=15, right_ctx=15, dct_nb=16):
    """

    :param X: matrix of acoustic frames
    :param left_ctx: left context of the frame to consider (given in number of frames)
    :param right_ctx: right context of the frame to consider (given in number of frames)
    :param dct_nb: number of DCT coefficient to keep for dimensionality reduction
    :return: matrix of traps features (in rows)
    """
    X = framing(X, win_size=left_ctx + 1 + right_ctx).transpose(0, 2, 1)
    hamming_dct = (dct_basis(dct_nb, left_ctx + right_ctx + 1) * np.hamming(left_ctx + right_ctx + 1)).T.astype(
        "float32")
    return np.dot(X.reshape(-1, hamming_dct.shape[0]), hamming_dct).reshape(X.shape[0], -1)


def get_context(X, left_ctx=7, right_ctx=7, apply_hamming=False):
    """

    :param X:  matrix of acoustic frames
    :param left_ctx: left context of the frame to consider (given in number of frames)
    :param right_ctx: right context of the frame to consider (given in number of frames)
    :param apply_hamming: boolean, if True, multiply by a temporal hamming window
    :return: a matrix of frames concatenated with their left and right context
    """
    X = framing(X, win_size=left_ctx + 1 + right_ctx).reshape(-1, (left_ctx + 1 + right_ctx) * X.shape[1])
    return X
