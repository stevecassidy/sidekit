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

import numpy
import scipy
from scipy.fftpack.realtransforms import dct
from sidekit.frontend.vad import pre_emphasis
from sidekit.frontend.io import *
from sidekit.frontend.normfeat import *
from sidekit.frontend.features import *
import numpy.matlib

PARAM_TYPE = numpy.float32

__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2016 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def hz2mel(f, htk=True):
    """Convert an array of frequency in Hz into mel.
    
    :param f: frequency to convert
    
    :return: the equivalence on the mel scale.
    """
    if htk:
        return 2595 * numpy.log10(1 + f / 700.)
    else:
        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        f_0 = 0.
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log2(6.4) / 27)

        linpts = f < brkfrq

        z = numpy.zeros_like(f)

        # fill in parts separately
        z[linpts] = (f[linpts] - f_0) / f_sp
        z[~linpts] = brkpt + (numpy.log2(f[~linpts] / brkfrq)) / numpy.log(logstep)

        return z


def mel2hz(m, htk=True):
    """Convert an array of mel values in Hz.
    
    :param m: ndarray of frequencies to convert in Hz.
    
    :return: the equivalent values in Hertz.
    """
    if htk:
        return 700. * (10**(z / 2595.) - 1)
    else:
        f_0 = 0
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log2(6.4) / 27)

        linpts = (z < brkpt)

        f = numpy.zeros_like(z)

        # fill in parts separately
        f[linpts] = f_0 + f_sp * z[linpts]
        f[~linpts] = brkfrq * numpy.exp(numpy.log2(logstep) * (z[~linpts] - brkpt))

        return f


def hz2bark(f):
    """
    Convert frequencies (Hertz) to Bark frequencies

    :param f: the input frequency
    :return:
    """
    return 6. * numpy.arcsinh(f / 600.)


def bark2hz(z):
    """
    Converts frequencies Bark to Hertz (Hz)

    :param z:
    :return:
    """
    return 600. * numpy.sinh(z / 6.)


def compute_delta(features,
                  win=3,
                  method='filter',
                  filt=numpy.array([.25, .5, .25, 0, -.25, -.5, -.25])):
    """features is a 2D-ndarray  each row of features is a a frame
    
    :param features: the feature frames to compute the delta coefficients
    :param win: parameter that set the length of the computation window.
            The size of the window is (win x 2) + 1
    :param method: method used to compute the delta coefficients
        can be diff or filter
    :param filt: definition of the filter to use in "filter" mode, default one
        is similar to SPRO4:  filt=numpy.array([.2, .1, 0, -.1, -.2])
        
    :return: the delta coefficients computed on the original features.
    """
    # First and last features are appended to the begining and the end of the 
    # stream to avoid border effect
    x = numpy.zeros((features.shape[0] + 2 * win, features.shape[1]), dtype=PARAM_TYPE)
    x[:win, :] = features[0, :]
    x[win:-win, :] = features
    x[-win:, :] = features[-1, :]

    delta = numpy.zeros(x.shape, dtype=PARAM_TYPE)

    if method == 'diff':
        filt = numpy.zeros(2 * win + 1, dtype=PARAM_TYPE)
        filt[0] = -1
        filt[-1] = 1

    for i in range(features.shape[1]):
        delta[:, i] = numpy.convolve(features[:, i], filt)

    return delta[win:-win, :]


def pca_dct(cep, left_ctx=12, right_ctx=12, p=None):
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
    :param p: a PCA matrix trained on a developpment set to reduce the
       dimension of the features. P is a portait matrix
    """
    y = numpy.r_[numpy.resize(cep[0, :], (left_ctx, cep.shape[1])),
                 cep,
                 numpy.resize(cep[-1, :], (right_ctx, cep.shape[1]))]

    ceps = framing(y, win_size=left_ctx + 1 + right_ctx).transpose(0, 2, 1)
    dct_temp = (dct_basis(left_ctx + 1 + right_ctx, left_ctx + 1 + right_ctx)).T
    if p is None:
        p = numpy.eye(dct_temp.shape[0] * cep.shape[1], dtype=PARAM_TYPE)
    return (numpy.dot(ceps.reshape(-1, dct_temp.shape[0]),
                      dct_temp).reshape(ceps.shape[0], -1)).dot(p)


def shifted_delta_cepstral(cep, d=1, p=3, k=7):
    """
    Compute the Shifted-Delta-Cepstral features for language identification
    
    :param cep: matrix of feature, 1 vector per line
    :param d: represents the time advance and delay for the delta computation
    :param k: number of delta-cepstral blocks whose delta-cepstral 
       coefficients are stacked to form the final feature vector
    :param p: time shift between consecutive blocks.
    
    return: cepstral coefficient concatenated with shifted deltas
    """

    y = numpy.r_[numpy.resize(cep[0, :], (d, cep.shape[1])),
                 cep,
                 numpy.resize(cep[-1, :], (k * 3 + d, cep.shape[1]))]

    delta = compute_delta(y, win=d, method='diff')

    sdc = numpy.empty((cep.shape[0], cep.shape[1] * k))

    idx = numpy.zeros(len(sdc), dtype='bool')
    for ii in range(k):
        idx[d + ii * p] = True
    for ff in range(len(cep)):
        sdc[ff, :] = delta[idx, :].reshape(1, -1)
        idx = numpy.roll(idx, 1)
    return numpy.hstack((cep, sdc))


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
    frequences = numpy.zeros(nfilt + 2, dtype=PARAM_TYPE)
    if nlogfilt == 0:
        linsc = (maxfreq - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt + 2] = lowfreq + numpy.arange(nlinfilt + 2) * linsc
    elif nlinfilt == 0:
        low_mel = hz2mel(lowfreq)
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2)
        # mels[nlinfilt:]
        melsc = (max_mel - low_mel) / (nfilt + 1)
        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences = mel2hz(mels)
    else:
        # Compute linear filters on [0;1000Hz]
        linsc = (min([midfreq, maxfreq]) - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
        # Compute log-linear filters on [1000;maxfreq]
        low_mel = hz2mel(min([1000, maxfreq]))
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2, dtype=PARAM_TYPE)
        melsc = (max_mel - low_mel) / (nlogfilt + 1)

        # Verify that mel2hz(melsc)>linsc
        while mel2hz(melsc) < linsc:
            # in this case, we add a linear filter
            nlinfilt += 1
            nlogfilt -= 1
            frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
            low_mel = hz2mel(frequences[nlinfilt - 1] + 2 * linsc)
            max_mel = hz2mel(maxfreq)
            mels = numpy.zeros(nlogfilt + 2, dtype=PARAM_TYPE)
            melsc = (max_mel - low_mel) / (nlogfilt + 1)

        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences[nlinfilt:] = mel2hz(mels)

    heights = 2. / (frequences[2:] - frequences[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, int(numpy.floor(nfft / 2)) + 1), dtype=PARAM_TYPE)
    # FFT bins (in Hz)
    n_frequences = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = frequences[i]
        cen = frequences[i + 1]
        hi = frequences[i + 2]

        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1, numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        left_slope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1,
                           min(numpy.floor(hi * nfft / fs) + 1, nfft), dtype=numpy.int)
        right_slope = heights[i] / (hi - cen)
        fbank[i][lid] = left_slope * (n_frequences[lid] - low)
        fbank[i][rid[:-1]] = right_slope * (hi - n_frequences[rid[:-1]])

    return fbank, frequences


def mel_filter_bank(fs, nfft, lowfreq, maxfreq, widest_nlogfilt, widest_lowfreq, widest_maxfreq,):
    """Compute triangular filterbank for cepstral coefficient computation.

    :param fs: sampling frequency of the original signal.
    :param nfft: number of points for the Fourier Transform
    :param lowfreq: lower limit of the frequency band filtered
    :param maxfreq: higher limit of the frequency band filtered
    :param widest_nlogfilt: number of log filters
    :param widest_lowfreq: lower frequency of the filter bank
    :param widest_maxfreq: higher frequency of the filter bank
    :param widest_maxfreq: higher frequency of the filter bank

    :return: the filter bank and the central frequencies of each filter
    """

    # ------------------------
    # Compute the filter bank
    # ------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    widest_freqs = numpy.zeros(widest_nlogfilt + 2, dtype=PARAM_TYPE)
    low_mel = hz2mel(widest_lowfreq)
    max_mel = hz2mel(widest_maxfreq)
    mels = numpy.zeros(widest_nlogfilt+2)
    melsc = (max_mel - low_mel) / (widest_nlogfilt + 1)
    mels[:widest_nlogfilt + 2] = low_mel + numpy.arange(widest_nlogfilt + 2) * melsc
    # Back to the frequency domain
    widest_freqs = mel2hz(mels)

    # Select filters in the narrow band
    sub_band_freqs = numpy.array([fr for fr in widest_freqs if lowfreq <= fr <= maxfreq], dtype=PARAM_TYPE)

    heights = 2./(sub_band_freqs[2:] - sub_band_freqs[0:-2])
    nfilt = sub_band_freqs.shape[0] - 2

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, numpy.floor(nfft/2)+1), dtype=PARAM_TYPE)
    # FFT bins (in Hz)
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = sub_band_freqs[i]
        cen = sub_band_freqs[i+1]
        hi = sub_band_freqs[i+2]
        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1, numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        left_slope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1, min(numpy.floor(hi * nfft / fs) + 1,
                                                                 nfft), dtype=numpy.int)
        right_slope = heights[i] / (hi - cen)
        fbank[i][lid] = left_slope * (nfreqs[lid] - low)
        fbank[i][rid[:-1]] = right_slope * (hi - nfreqs[rid[:-1]])

    return fbank, sub_band_freqs


def mfcc(input_sig,
         lowfreq=100, maxfreq=8000,
         nlinfilt=0, nlogfilt=24,
         nwin=0.025,
         fs=16000,
         nceps=13,
         shift=0.01,
         get_spec=False,
         get_mspec=False,
         prefac=0.97):
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
    :param prefac: pre-emphasis filter value

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

    # Prepare the signal to be processed on a sliding window.
    # We first compute the overlap of frames and cut the signal in frames of length nwin
    # overlaping by "overlap" samples
    window_length = int(round(nwin * fs))
    overlap = window_length - int(shift * fs)
    framed = framing(input_sig, window_length, win_shift=window_length-overlap).copy()

    # Pre-emphasis filtering is applied after framing to be consistent with stream processing
    framed = pre_emphasis(framed, prefac)

    l = framed.shape[0]
    n_fft = 2 ** int(numpy.ceil(numpy.log2(window_length)))
    ham = numpy.hamming(window_length)
    spec = numpy.ones((l, int(n_fft / 2) + 1), dtype=PARAM_TYPE)
    log_energy = numpy.log((framed**2).sum(axis=1))
    dec = 500000
    start = 0
    stop = min(dec, l)
    while start < l:
        aham = framed[start:stop, :] * ham
        mag = numpy.fft.rfft(aham, n_fft, axis=-1)
        spec[start:stop, :] = mag.real**2 + mag.imag**2
        start = stop
        stop = min(stop + dec, l)
    del framed

    # Filter the spectrum through the triangle filter-bank
    fbank = trfbank(fs, n_fft, lowfreq, maxfreq, nlinfilt, nlogfilt)[0]

    mspec = numpy.log(numpy.dot(spec, fbank.T))   # A tester avec log10 et log

    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    # The C0 term is removed as it is the constant term
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, 1:nceps + 1]
    lst = list()
    lst.append(ceps)
    lst.append(log_energy)
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


"""
PLP IMPLEMENTATION
"""
def powspec(input_sig,
            fs=8000,
            win_time=0.025,
            shift=0.01,
            prefac=0.97):
    """

    :param input_sig:
    :param fs:
    :param win_time:
    :param shift:
    :return:
    """
    window_length = int(round(win_time * fs))
    overlap = window_length - int(shift * fs)
    framed = framing(input_sig, window_length, win_shift=window_length-overlap).copy()

    # Pre-emphasis filtering is applied after framing to be consistent with stream processing
    framed = pre_emphasis(framed, prefac)

    l = framed.shape[0]
    nfft = 2 ** int(numpy.ceil(numpy.log2(window_length)))
    # Windowing has been changed to hanning which is supposed to have less noisy sidelobes
    # ham = numpy.hamming(window_length)
    window = numpy.hanning(window_length)

    spec = numpy.ones((l, int(nfft / 2) + 1), dtype=PARAM_TYPE)
    log_energy = numpy.log((framed**2).sum(axis=1))
    dec = 500000
    start = 0
    stop = min(dec, l)
    while start < l:
        ahan = framed[start:stop, :] * window
        mag = numpy.fft.rfft(ahan, nfft, axis=-1)
        spec[start:stop, :] = mag.real**2 + mag.imag**2
        start = stop
        stop = min(stop + dec, l)

    return spec, log_energy


def fft2barkmx(nfft, sr, nfilts=0, width=1., minfreq=0., maxfreq=8000):
    """
    Generate a matrix of weights to combine FFT bins into Bark
    bins.  nfft defines the source FFT size at sampling rate sr.
    Optional nfilts specifies the number of output bands required
    (else one per bark), and width is the constant width of each
    band in Bark (default 1).
    While wts has nfft columns, the second half are all zero.
    Hence, Bark spectrum is fft2barkmx(nfft,sr)*abs(fft(xincols,nfft));
    2004-09-05  dpwe@ee.columbia.edu  based on rastamat/audspec.m

    :param nfft: the source FFT size at sampling rate sr
    :param sr: sampling rate
    :param nfilts: number of output bands required
    :param width: constant width of each band in Bark (default 1)
    :param minfreq:
    :param maxfreq:
    :return: a matrix of weights to combine FFT bins into Bark bins
    """
    #maxfreq = min(maxfreq, sr / 2.)

    #min_bark = hz2bark(minfreq)
    #nyqbark = hz2bark(maxfreq) - min_bark
    #if nfilts == 0:
    #    nfilts = numpy.ceil(nyqbark) + 1

    #wts = numpy.zeros((nfilts, nfft))

    # bark per filt
    #step_barks = nyqbark / (nfilts - 1)

    # Frequency of each FFT bin in Bark
    #binbarks = hz2bark(numpy.arange(nfft / 2) * sr / nfft)

    #for i in range(1, nfilts + 1):
    #    f_bark_mid = min_bark + (i - 1) * step_barks;
    #    # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
    #    lof = (binbarks - f_bark_mid - 0.5)
    #    hif = (binbarks - f_bark_mid + 0.5)
    #    #LIGNE A VERIFIER...
    #    wts[i, :nfft/2+1] = 10**(min(0, min([hif; -2.5*lof]) / width))


def fft2melmx(nfft,
              sr=8000,
              nfilts=0,
              width=1.,
              minfrq=0,
              maxfrq=4000,
              htkmel=False,
              constamp=False):
    """
    Generate a matrix of weights to combine FFT bins into Mel
    bins.  nfft defines the source FFT size at sampling rate sr.
    Optional nfilts specifies the number of output bands required
    (else one per "mel/width"), and width is the constant width of each
    band relative to standard Mel (default 1).
    While wts has nfft columns, the second half are all zero.
    Hence, Mel spectrum is fft2melmx(nfft,sr)*abs(fft(xincols,nfft));
    minfrq is the frequency (in Hz) of the lowest band edge;
    default is 0, but 133.33 is a common standard (to skip LF).
    maxfrq is frequency in Hz of upper edge; default sr/2.
    You can exactly duplicate the mel matrix in Slaney's mfcc.m
    as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
    htkmel=1 means use HTK's version of the mel curve, not Slaney's.
    constamp=1 means make integration windows peak at 1, not sum to 1.
    frqs returns bin center frqs.

    % 2004-09-05  dpwe@ee.columbia.edu  based on fft2barkmx

    :param nfft:
    :param sr:
    :param nfilts:
    :param width:
    :param minfrq:
    :param maxfrq:
    :param htkmel:
    :param constamp:
    :return:
    """
    if nfilts == 0:
        nfilts = numpy.ceil(hz2mel(maxfrq, htkmel) / 2.)

    wts = numpy.zeros((nfilts, nfft))

    # Center freqs of each FFT bin
    fftfrqs = numpy.arange(nfft / 2) / nfft * sr

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfrq, htkmel)
    maxmel = hz2mel(maxfrq, htkmel)
    binfrqs = mel2hz(minmel +  numpy.arange(nfilts + 1) / (nfilts + 1) * (maxmel - minmel), htkmel)

    binbin = numpy.around(binfrqs / sr * (nfft - 1))

    for i in range(nfilts):
        fs = binfrqs[i + numpy.arange(3, dtype=int)]
        # scale by width
        fs = fs[1] + width * (fs - fs[1])
        # lower and upper slopes for all bins
        loslope = (fftfrqs - fs[0]) / (fs[1] - fs[2])
        hislope = (fs[2] - fftfrqs)/(fs[2] - fs[1])
        # .. then intersect them with each other and zero
        # wts(i,:) = 2/(fs(3)-fs(1))*max(0,min(loslope, hislope));
        wts[i, 1 + numpy.arange(nfft/2)] = max(0,min(loslope, hislope))

    if not constamp:
        # Slaney-style mel is scaled to be approx constant E per channel
        wts = numpy.diag(2. / (binfrqs[2 + numpy.arange(nfilts)] - binfrqs[numpy.arange(nfilts)])) * wts

    # Make sure 2nd half of FFT is zero
    wts[:, nfft / 2 + 2: nfft] = 0

    return wts, binfrqs


def audspec(pspectrum,
            sr=16000,
            nfilts=None,
            fbtype='bark',
            minfreq=0,
            maxfreq=8000,
            sumpower=True,
            bwidth=1.):
    """

    :param pspectrum:
    :param sr:
    :param nfilts:
    :param fbtype:
    :param minfreq:
    :param maxfreq:
    :param sumpower:
    :param bwidth:
    :return:
    """
    if nfilts is None:
        nfilts = numpy.ceiling(hz2bark(sr / 2)) + 1
    if not sr == 16000:
        maxfreq = numpy.min(sr / 2, maxfreq)

    nframes, nfreqs = pspectrum.shape

    nfft = (nfreqs -1 ) * 2

    if fbtype == 'bark':
        wts = fft2barkmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'mel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'htkmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 1)
    elif fbtype == 'fcmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 0)
    else:
        print('fbtype {} not recognized'.format(fbtype))

    wts = wts[:, :nfreqs]

    # Integrate FFT bins into Mel bins, in abs or abs^2 domains:
    if sumpower:
        aspectrum = wts * pspectrum
    else:
        aspectrum = (wts * numpy.sqrt(pspectrum))**2

    return aspectrum, wts


def postaud(x, fmax, fbtype='bark', broaden=False):
    """
    do loudness equalization and cube root compression

    :param x:
    :param fmax:
    :param fbtype:
    :param broaden:
    :return:
    """
    nframes, nbands = x.shape

    # equal loundness weights stolen from rasta code
    eql = numpy.array([0.000479, 0.005949, 0.021117, 0.044806, 0.073345,
                       0.104417, 0.137717, 0.174255, 0.215590, 0.263260,
                       0.318302, 0.380844, 0.449798, 0.522813, 0.596597])

    # Include frequency points at extremes, discard later
    nfpts = nbands + 2 * broaden

    if fbtype == 'bark':
        bandcfhz = bark2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'mel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'htkmel' or fbtype == 'fcmel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2mel(fmax,1), num=nfpts),1)
    else:
        print('unknown fbtype {}'.format(fbtype))

    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[(1 + broaden):(nfpts - broaden)]

    # Hynek's magic equal-loudness-curve formula
    fsq = bandcfhz**2
    ftmp = fsq + 1.6e5
    eql = ((fsq / ftmp)**2) * ((fsq + 1.44e6) / (fsq + 9.61e6))

    # weight the critical bands
    z = numpy.matlib.repmat(eql.T,1,nframes) * x

    # cube root compress
    z = z**.33

    # replicate first and last band (because they are unreliable as calculated)
    if broaden:
      y = z[numpy.hstack((1,numpy.arange(nbands), nbands - 1)), :]
    else:
      y = z[numpy.hstack((1,numpy.arange(1, nbands - 1), nbands - 2)), :]

    return y,eql


def dolpc(x, model_order=8):
    """
    compute autoregressive model from spectral magnitude samples

    :param x:
    :param model_order:
    :return:
    """
    #nframes, nbands = x.shape

    # Calculate autocorrelation
    #r = real(ifft([x;x([(nbands-1):-1:2],:)]))
    # First half only
    #r = r(1:nbands,:);

    # Find LPC coeffs by durbin
    #[y,e] = levinson(r, modelorder);

    # Normalize each poly by gain
    #y = y'./repmat(e',(modelorder+1),1);

    #return y


def lpc2cep(a, nout):
    """
    Convert the LPC 'a' coefficients in each column of lpcas
    into frames of cepstra.
    nout is number of cepstra to produce, defaults to size(lpcas,1)
    2003-04-11 dpwe@ee.columbia.edu

    :param a:
    :param nout:
    :return:
    """
    nin, ncol = a.shape

    order = nin - 1

    if nout is None:
        nout = order + 1

    c = numpy.zeros((nout, ncol))

    # Code copied from HSigP.c: LPC2Cepstrum
    c[0, :] = -numpy.log2(a[0, :])

    # Renormalize lpc A coeffs
    a = a / numpy.matlib.repmat(a[0, :], nin, 1)

    for n in range(1, nout):
        sum = 0
        for m in range(1, n):
            sum += (n - m) * a[m, :] * c[n - m + 1, :]
        c[n, :] = -(a[n, :] + sum / (n-1))

    return c


def lpc2spec(lpcas, nout=17):
    """
    Convert LPC coeffs back into spectra
    nout is number of freq channels, default 17 (i.e. for 8 kHz)

    :param lpcas:
    :param nout:
    :return:
    """

    #[rows, cols] = lpcas.shape
    #order = rows - 1

    #gg = lpcas[1, :]
    #aa = lpcas / numpy.matlab.repmat(gg,rows,1)

    # Calculate the actual z-plane polyvals: nout points around unit circle
    #zz = numpy.exp((-j * [0:(nout-1)]' * pi / (nout - 1)) * [0:order])

    # Actual polyvals, in power (mag^2)
    #features =  ((1./abs(zz*aa)).^2)./repmat(gg,nout,1);

    #F = numpy.zeros((cols, numpy.floor(rows / 2)))
    #M = F;

    #for c in range(cols):
    #    aaa = aa(:,c);
    #    rr = roots(aaa');
    #    ff = angle(rr');

    #    zz = exp(j*ff'*[0:(length(aaa)-1)]);
    #    mags = sqrt(((1./abs(zz*aaa)).^2)/gg(c))';

    #    [dummy, ix] = sort(ff);
    #    keep = ff(ix) > 0;
    #    ix = ix(keep);
    #    F(c,1:length(ix)) = ff(ix);
    #    M(c,1:length(ix)) = mags(ix);

    #return features, F, M


def spec2cep(spec, ncep=13, type=2):
    """
    Calculate cepstra from spectral samples (in columns of spec)
    Return ncep cepstral rows (defaults to 9)
    This one does type II dct, or type I if type is specified as 1
    dctm returns the DCT matrix that spec was multiplied by to give cep.

    :param spec:
    :param ncep:
    :param type:
    :return:
    """
    nrow, ncol = spec.shape

    # Make the DCT matrix
    dctm = numpy.zeros(ncep, nrow);
    #if type == 2 || type == 3
    #    # this is the orthogonal one, the one you want
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[1:2:(2*nrow-1)]/(2*nrow)*pi) * sqrt(2/nrow);

    #    if type == 2
    #        # make it unitary! (but not for HTK type 3)
    #        dctm(1,:) = dctm(1,:)/sqrt(2);

    #elif type == 4:  # type 1 with implicit repeating of first, last bins
    #    """
    #    Deep in the heart of the rasta/feacalc code, there is the logic
    #    that the first and last auditory bands extend beyond the edge of
    #    the actual spectra, and they are thus copied from their neighbors.
    #    Normally, we just ignore those bands and take the 19 in the middle,
    #    but when feacalc calculates mfccs, it actually takes the cepstrum
    #    over the spectrum *including* the repeated bins at each end.
    #    Here, we simulate 'repeating' the bins and an nrow+2-length
    #    spectrum by adding in extra DCT weight to the first and last
    #    bins.
    #    """
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[1:nrow]/(nrow+1)*pi) * 2;
    #        # Add in edge points at ends (includes fixup scale)
    #        dctm(i,1) = dctm(i,1) + 1;
    #        dctm(i,nrow) = dctm(i,nrow) + ((-1)^(i-1));

    #   dctm = dctm / (2*(nrow+1));
    #else % dpwe type 1 - same as old spec2cep that expanded & used fft
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[0:(nrow-1)]/(nrow-1)*pi) * 2 / (2*(nrow-1));
    #    dctm(:,[1 nrow]) = dctm(:, [1 nrow])/2;

    #cep = dctm*log(spec);
    return None, None, None

def lifter(x, lift=0.6, invs=False):
    """
    Apply lifter to matrix of cepstra (one per column)
    lift = exponent of x i^n liftering
    or, as a negative integer, the length of HTK-style sin-curve liftering.
    If inverse == 1 (default 0), undo the liftering.

    :param x:
    :param lift:
    :param invs:
    :return:
    """
    nfrm, ncep = x.shape

    #if lift == 0:
    #  y = x;
    #else:
    #  if lift > 0:
    #    if lift > 10:
    #      print('Unlikely lift exponent of {} did you mean -ve?'.format(num2str(lift)))

    #    liftwts = [1, ([1:(ncep-1)].^lift)];
    #  elseif lift < 0
    #    % Hack to support HTK liftering
    #    L = -lift;
    #    if (L ~= round(L))
    #      disp(['HTK liftering value ', num2str(L),' must be integer']);
    #    end
    #    liftwts = [1, (1+L/2*sin([1:(ncep-1)]*pi/L))];
    #  end

    #  if (invs)
    #    liftwts = 1./liftwts;
    #  end

    #  y = diag(liftwts)*x;

    return None

def plp(input_sig,
        fs=8000,
        rasta=True,
        model_order=8):
    """
    output is matrix of features, row = feature, col = frame

% fs is sampling rate of samples, defaults to 8000
% dorasta defaults to 1; if 0, just calculate PLP
% modelorder is order of PLP model, defaults to 8.  0 -> no PLP

    :param input_sig:
    :param fs: sampling rate of samples default is 8000
    :param rasta: default is True, if False, juste compute PLP
    :param model_order: order of the PLP model, default is 8, 0 means no PLP

    :return: matrix of features, row = features, column are frames
    """

    # add miniscule amount of noise
    # input_sig = input_sig + randn(size(input_sig))*0.0001;

    # first compute power spectrum
    pspectrum = powspec(input_sig, fs)

    # next group to critical bands
    aspectrum = audspec(pspectrum, fs)
    nbands = aspectrum.shape[0]

    if rasta:
        # put in log domain
        nl_aspectrum = numpy.log2(aspectrum)

        #  next do rasta filtering
        ras_nl_aspectrum = rasta_filt(nl_aspectrum)

        # do inverse log
        aspectrum = numpy.exp(ras_nl_aspectrum)

    # do final auditory compressions
    postspectrum = postaud(aspectrum, fs / 2.)

    if model_order > 0:

        # LPC analysis
        lpcas = dolpc(postspectrum, model_order)

        # convert lpc to cepstra
        cepstra = lpc2cep(lpcas, model_order + 1)

        # .. or to spectra
        spectra, F, M = lpc2spec(lpcas, nbands)

    else:

        # No LPC smoothing of spectrum
        spectra = postspectrum
        cepstra = spec2cep(spectra)

    cepstra = lifter(cepstra, 0.6)

    return cepstra, spectra, pspectrum, lpcas, F, M

def framing(sig, win_size, win_shift=1, context=(0, 0), pad='zeros'):
    """
    :param sig: input signal, can be mono or multi dimensional
    :param win_size: size of the window in term of samples
    :param win_shift: shift of the sliding window in terme of samples
    :param context: tuple of left and right context
    :param pad: can be zeros or edge
    """
    dsize = sig.dtype.itemsize
    if sig.ndim == 1:
        sig = sig[:, numpy.newaxis]
    # Manage padding
    c = (context, ) + (sig.ndim - 1) * ((0, 0), )
    _win_size = win_size + sum(context)
    shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
    strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    if pad == 'zeros':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                                  shape=shape,
                                                  strides=strides).squeeze()
    elif pad == 'edge':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'edge'),
                                                  shape=shape,
                                                  strides=strides).squeeze()


def dct_basis(nbasis, length):
    """
    :param nbasis: number of CT coefficients to keep
    :param length: length of the matrix to process
    :return: a basis of DCT coefficients
    """
    return scipy.fftpack.idct(numpy.eye(nbasis, length), norm='ortho')
