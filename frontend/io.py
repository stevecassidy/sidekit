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

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""
import numpy as np
import struct
import math
import os
import decimal
import wave
import logging
import audioop
from scipy.io import wavfile

from sidekit.sidekit_io import *


__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


@check_path_existance
def write_pcm(data, outputFileName):
    """Write signal to single channel PCM 16 bits
    
    :param data: audio signal to write in a RAW PCM file.
    :param outputFileName: name of the file to write
    """
    with open(outputFileName, 'w') as of:
        data *= 16384
        of.write(struct.pack('<' + 'h' * data.shape[0], *data))


def read_pcm(inputFileName):
    """Read signal from single channel PCM 16 bits

    :param inputFileName: name of the PCM file to read.
    
    :return: the audio signal read from the file in a ndarray.
    """
    with open(inputFileName, 'rb') as f:
        f.seek(0, 2)  # Go to te end of the file
        # get the sample count
        sampleCount = int(f.tell() / 2)
        f.seek(0, 0)  # got to the begining of the file
        data = np.asarray(struct.unpack('<' + 'h' * sampleCount, f.read()))
    return data/32768.0 


def read_wav(inputFileName):
    """Read signal from a wave file
    
    :param inputFileName: name of the PCM file to read.
    
    :return: the audio signal read from the file in a ndarray.
    """
    framerate, sig = wavfile.read(inputFileName)
    return sig, framerate


def pcmu2lin(p, s=4004.189931):
    """Convert Mu-law PCM to linear X=(P,S)
    lin = pcmu2lin(pcmu) where pcmu contains a vector
    of mu-law values in the range 0 to 255.
    No checking is performed to see that numbers are in this range.

    Output values are divided by the scale factor s:

        s		Output Range
        1		+-8031	(integer values)
        4004.2	+-2.005649 (default)
        8031		+-1
        8159		+-0.9843118 (+-1 nominal full scale)

    The default scaling factor 4004.189931 is equal to
    sqrt((2207^2 + 5215^2)/2) this follows ITU standard G.711.
    The sine wave with PCM-Mu values [158 139 139 158 30 11 11 30]
    has a mean square value of unity corresponding to 0 dBm0.
    :param p: input signal encoded in PCM mu-law to convert
    :param s: conversion value from mu-scale oto linear scale
    """
    t = 4 / s
    m = 15 - (p % 16)
    q = np.floor(p // 128)
    e = (127 - p - m + 128 * q) / 16
    x = (m + 16.5) * np.power(2, e) - 16.5
    z = (q - 0.5) * x * t
    return z


def read_sph(inputFileName, mode='p'):
    """
    Read a SPHERE audio file

    :param inputFileName: name of the file to read
    :param mode: specifies the following (\* =default)
    
    .. note::
    
        - Scaling:
        
            - 's'    Auto scale to make data peak = +-1 (use with caution if reading in chunks)
            - 'r'    Raw unscaled data (integer values)
            - 'p'    Scaled to make +-1 equal full scale
            - 'o'    Scale to bin centre rather than bin edge (e.g. 127 rather than 127.5 for 8 bit values,
                     can be combined with n+p,r,s modes)
            - 'n'    Scale to negative peak rather than positive peak (e.g. 128.5 rather than 127.5 for 8 bit values,
                     can be combined with o+p,r,s modes)

        - Format
       
           - 'l'    Little endian data (Intel,DEC) (overrides indication in file)
           - 'b'    Big endian data (non Intel/DEC) (overrides indication in file)

       - File I/O
       
           - 'f'    Do not close file on exit
           - 'd'    Look in data directory: voicebox('dir_data')
           - 'w'    Also read the annotation file \*.wrd if present (as in TIMIT)
           - 't'    Also read the phonetic transcription file \*.phn if present (as in TIMIT)

        - NMAX     maximum number of samples to read (or -1 for unlimited [default])
        - NSKIP    number of samples to skip from start of file (or -1 to continue from previous read when FFX
                   is given instead of FILENAME [default])

    :return: a tupple such that (Y, FS)
    
    .. note::
    
        - Y data matrix of dimension (samples,channels)
        - FS         sample frequency in Hz
        - WRD{\*,2}  cell array with word annotations: WRD{\*,:)={[t_start t_end],'text'} where times are in seconds
                     only present if 'w' option is given
        - PHN{\*,2}  cell array with phoneme annotations: PHN{\*,:)={[t_start	t_end],'phoneme'} where times
                     are in seconds only present if 't' option is present
        - FFX        Cell array containing

            1. filename
            2. header information
        
            1. first header field name
            2. first header field value
            3. format string (e.g. NIST_1A)
            4. 
                1. file id
                2. current position in file
                3. dataoff    byte offset in file to start of data
                4. order  byte order (l or b)
                5. nsamp    number of samples
                6. number of channels
                7. nbytes    bytes per data value
                8. bits    number of bits of precision
                9. fs	sample frequency
                10. min value
                11. max value
                12. coding 0=PCM,1=uLAW + 0=no compression, 0=shorten,20=wavpack,30=shortpack
                13. file not yet decompressed
                
            5. temporary filename

    If no output parameters are specified,
    header information will be printed.
    The code to decode shorten-encoded files, is 
    not yet released with this toolkit.
    """
    codings = dict([('pcm', 1), ('ulaw', 2)])
    compressions = dict([(',embedded-shorten-', 1),
                         (',embedded-wavpack-', 2),
                         (',embedded-shortpack-', 3)])
    BYTEORDER = 'l'
    endianess = dict([('l', '<'), ('b', '>')])

    if not mode == 'p':
        mode = [mode, 'p']
    k = list((m >= 'p') & (m <= 's') for m in mode)
    # scale to input limits not output limits
    mno = all([m != 'o' for m in mode])
    sc = ''
    if k[0]:
        sc = mode[0]
    # Get byte order (little/big endian)
    if any([m == 'l' for m in mode]):
        BYTEORDER = 'l'
    elif any([m == 'b' for m in mode]):
        BYTEORDER = 'b'
    ffx = ['', '', '', '', '']

    if isinstance(inputFileName, str):
        if os.path.exists(inputFileName):
            fid = open(inputFileName, 'rb')
        elif os.path.exists("".join((inputFileName, '.sph'))):
            inputFileName = "".join((inputFileName, '.sph'))
            fid = open(inputFileName, 'rb')
        else:
            pass  # TODO: RAISE an exception
        ffx[0] = inputFileName
    elif not isinstance(inputFileName, str):
        ffx = inputFileName
    else:
        fid = inputFileName

    # Read the header
    if ffx[3] == '':
        fid.seek(0, 0)  # go to the begining of the file
        l1 = fid.readline().decode("utf-8")
        l2 = fid.readline().decode("utf-8")
        if not (l1 == 'NIST_1A\n') & (l2 == '   1024\n'):
            logging.warning('File does not begin with a SPHERE header')
        ffx[2] = l1.rstrip()
        hlen = int(l2[3:7])
        hdr = {}
        while True:  # Read the header and fill a dictionary
            st = fid.readline().decode("utf-8").rstrip()
            if st[0] != ';':
                elt = st.split(' ')
                if elt[0] == 'end_head':
                    break
                if elt[1][0] != '-':
                    logging.warning('Missing ''-'' in SPHERE header')
                    break
                if elt[1][1] == 's':
                    hdr[elt[0]] = elt[2]
                elif elt[1][1] == 'i':
                    hdr[elt[0]] = int(elt[2])
                else:
                    hdr[elt[0]] = float(elt[2])

        if 'sample_byte_format' in list(hdr.keys()):
            if hdr['sample_byte_format'][0] == '0':
                bord = 'l'
            else:
                bord = 'b'
            if (bord != BYTEORDER) & all([m != 'b' for m in mode]) \
                    & all([m != 'l' for m in mode]):
                BYTEORDER = bord

        icode = 0  # Get encoding, default is PCM
        if 'sample_coding' in list(hdr.keys()):
            icode = -1  # unknown code
            for coding in list(codings.keys()):
                if hdr['sample_coding'].startswith(coding):
                    # is the signal compressed
                    # if len(hdr['sample_coding']) > codings[coding]:
                    if len(hdr['sample_coding']) > len(coding):
                        for compression in list(compressions.keys()):
                            if hdr['sample_coding'].endswith(compression):
                                icode = 10 * compressions[compression] \
                                        + codings[coding] - 1
                                break
                    else:  # if the signal is not compressed
                        icode = codings[coding] - 1
                break

        # initialize info of the files with default values
        info = [fid, 0, hlen, ord(BYTEORDER), 0, 1, 2, 16, 1, 1, -1, icode]

        # Get existing info from the header
        if 'sample_count' in list(hdr.keys()):
            info[4] = hdr['sample_count']
        if not info[4]:  # if no info sample_count or zero
            # go to the end of the file
            fid.seek(0, 2)  # Go to te end of the file
            # get the sample count
            info[4] = int(math.floor((fid.tell() - info[2]) / (info[5] * info[6])))  # get the sample_count
        if 'channel_count' in list(hdr.keys()):
            info[5] = hdr['channel_count']
        if 'sample_n_bytes' in list(hdr.keys()):
            info[6] = hdr['sample_n_bytes']
        if 'sample_sig_bits' in list(hdr.keys()):
            info[7] = hdr['sample_sig_bits']
        if 'sample_rate' in list(hdr.keys()):
            info[8] = hdr['sample_rate']
        if 'sample_min' in list(hdr.keys()):
            info[9] = hdr['sample_min']
        if 'sample_max' in list(hdr.keys()):
            info[10] = hdr['sample_max']

        ffx[1] = hdr
        ffx[3] = info

    info = ffx[3]
    ksamples = info[4]
    if ksamples > 0:
        fid = info[0]
        if (icode >= 10) & (ffx[4] == ''):  # read compressed signal
            # need to use a script with SHORTEN
            raise Exception('compressed signal, need to unpack in a script with SHORTEN')
        info[1] = ksamples
        # use modes o and n to determine effective peak
        pk = 2 ** (8 * info[6] - 1) * (1 + (float(mno) / 2 - int(all([m != 'b'
                                                                      for m in
                                                                      mode]))) / 2 **
                                       info[7])
        fid.seek(1024)  # jump after the header
        nsamples = info[5] * ksamples
        if info[6] < 3:
            if info[6] < 2:
                logging.debug('Sphere i1 PCM')
                y = np.fromfile(fid, endianess[BYTEORDER]+"i1", -1)
                if info[11] % 10 == 1:
                    if y.shape[0] % 2:
                        y = np.frombuffer(audioop.ulaw2lin(
                                np.concatenate((y, np.zeros(1, 'int8'))), 2), 
                                np.int16)[:-1]/32768.
                    else:
                        y = np.frombuffer(audioop.ulaw2lin(y, 2), np.int16)/32768.
                    pk = 1.
                else:
                    #y -= 128
                    y = y - 128
            else:
                logging.debug('Sphere i2')
                y = np.fromfile(fid, endianess[BYTEORDER]+"i2", -1)
        else:  # non verifie
            if info[6] < 4:
                y = np.fromfile(fid, endianess[BYTEORDER]+"i1", -1)
                y = y.reshape(nsamples, 3).transpose()
                y = (np.dot(np.array([1, 256, 65536]), y) - (np.dot(y[2, :], 2 ** (-7)).astype(int) * 2 ** 24))
            else:
                y = np.fromfile(fid, endianess[BYTEORDER]+"i4", -1)

        if sc != 'r':
            if sc == 's':
                if info[9] > info[10]:
                    info[9] = np.min(y)
                    info[10] = np.max(y)
                sf = 1 / np.max(list(list(map(abs, info[9:11])), axis=0))
            else:
                sf = 1 / pk
            #y *= sf
            y = sf * y

        if info[5] > 1:
            y = y.reshape(ksamples, info[5])
    else:
        y = np.array([])
    if mode != 'f':
        fid.close()
        info[0] = -1
        if not ffx[4] == '':
            pass  # VERIFY SCRIPT, WHICH CASE IS HANDLED HERE
    return y, int(info[8])


def read_audio(inputFileName, fs=16000):
    """ Read a 1 or 2-channel audio file in SPHERE, WAVE or RAW PCM format.
    The format is determined from the file extension.
    
    :param inputFileName: name of the file to read from
    :param fs: sampling frequency in Hz, default is 16000
    
    :return: the signal as a numpy array and the sampling frequency
    """
    ext = os.path.splitext(inputFileName)[-1]
    if ext.lower() == '.sph':
        sig, fs = read_sph(inputFileName, 'p')
    elif ext.lower() == '.wav' or ext.lower() == '.wave':
        sig, fs = read_wav(inputFileName)
    elif ext.lower() == '.pcm' or ext.lower() == '.raw':
        sig = read_pcm(inputFileName)
    else:
        logging.warning('Unknown extension of audio file')
        sig = None
        fs = None
    return sig.astype(np.float32), fs


@check_path_existance
def write_label(label,
                outputFileName,
                selectedLabel='speech',
                framePerSecond=100):
    """Save labels in ALIZE format

    :param outputFileName: name of the file to write to
    :param label: label to write in the file given as a ndarray of boolean
    :param selectedLabel: label to write to the file. Default is 'speech'.
    :param framePerSecond: number of frame per seconds. Used to convert
            the frame number into time. Default is 100.
    """
    if label.shape[0] > 0:
        bits = label[:-1] ^ label[1:]
        # convert true value into a list of feature indexes
        # append 0 at the beginning of the list, append the last index to the list
        idx = [0] + (np.arange(len(bits))[bits] + 1).tolist() + [len(label)]
        fs = decimal.Decimal(1) / decimal.Decimal(framePerSecond)
        # for each pair of indexes (idx[i] and idx[i+1]), create a segment
        with open(outputFileName, 'w') as fid:
            for i in range(~label[0], len(idx) - 1, 2):
                fid.write('{} {} {}\n'.format(str(idx[i]*fs),
                                              str(idx[i + 1]*fs), selectedLabel))


def read_label(inputFileName, selectedLabel='speech', framePerSecond=100):
    """Read label file in ALIZE format

    :param inputFileName: the label file name
    :param selectedLabel: the label to return. Default is 'speech'.
    :param framePerSecond: number of frame per seconds. Used to convert 
            the frame number into time. Default is 100.

    :return: a logical array
    """
    with open(inputFileName) as f:
        segments = f.readlines()

    if len(segments) == 0:
        lbl = np.zeros(0).astype(bool)
    else:
        # initialize the length from the last segment's end
        foo1, stop, foo2 = segments[-1].rstrip().split()
        lbl = np.zeros(int(float(stop) * 100)).astype(bool)
    
        begin = np.zeros(len(segments))
        end = np.zeros(len(segments))
    
        for s in range(len(segments)):
            start, stop, label = segments[s].rstrip().split()
            if label == selectedLabel:
                begin[s] = int(round(float(start) * framePerSecond))
                end[s] = int(round(float(stop) * framePerSecond))
                lbl[begin[s]:end[s]] = True
    return lbl


def read_spro4(inputFileName,
               labelFileName="",
               selectedLabel="",
               framePerSecond=100):
    """Read a feature stream in SPRO4 format 
    
    :param inputFileName: name of the feature file to read from
    :param labelFileName: name of the label file to read if required.
        By Default, the method assumes no label to read from.    
    :param selectedLabel: label to select in the label file. Default is none.
    :param framePerSecond: number of frame per seconds. Used to convert 
            the frame number into time. Default is 0.
    
    :return: a sequence of features in a ndarray
    """
    with open(inputFileName, 'rb') as f:

        tmpS = struct.unpack("8c", f.read(8))
        S = ()
        for i in range(len(tmpS)):
            S = S + (tmpS[i].decode("utf-8"),)
        f.seek(0, 2)  # Go to te end of the file
        size = f.tell()  # get the position
        f.seek(0, 0)  # go back to the begining of the file
        headsize = 0

        if "".join(S) == '<header>':
            # swap empty header for general header the code need changing
            struct.unpack("19b", f.read(19))
            headsize = 19

        dim = struct.unpack("H", f.read(2))[0]
        struct.unpack("4b", f.read(4))
        struct.unpack("f", f.read(4))
        nframes = int(math.floor((size - 10 - headsize) / (4 * dim)))

        features = np.asarray(struct.unpack('f' * nframes * dim,
                                            f.read(4 * nframes * dim)))
        features.resize(nframes, dim)

    lbl = np.ones(np.shape(features)[0]).astype(bool)
    if not labelFileName == "":
        lbl = read_label(labelFileName, selectedLabel, framePerSecond)

    features = features[lbl, :]
    return features


def read_spro4_segment(inputFileName, start=0, end=None):
    """Read a segment from a stream in SPRO4 format. Return the features in the
    range start:end
    In case the start and end cannot be reached, the first or last feature are copied
    so that the length of the returned segment is always end-start
    
    :param inputFileName: name of the feature file to read from
    :param start: index of the first frame to read (start at zero)
    :param end: index of the last frame following the segment to read.
       end < 0 means that end is the value of the right_context to add 
       at the end of the file

    :return: a sequence of features in a ndarray of length end-start
    """
    with open(inputFileName, 'rb') as f:

        tmpS = struct.unpack("8c", f.read(8))
        S = ()
        for i in range(len(tmpS)):
            S = S + (tmpS[i].decode("utf-8"),)
        f.seek(0, 2)  # Go to te end of the file
        size = f.tell()  # get the position
        f.seek(0, 0)  # go back to the begining of the file
        headsize = 0

        if "".join(S) == '<header>':
            # swap empty header for general header the code need changing
            struct.unpack("19b", f.read(19))
            headsize = 19

        dim = struct.unpack("H", f.read(2))[0]
        struct.unpack("4b", f.read(4))
        struct.unpack("f", f.read(4))
        nframes = int(math.floor((size - 10 - headsize) / (4 * dim)))
        if end is None:
            end = nframes
        elif end < 0:
            end = nframes - end
            
        s, e = max(0, start), min(nframes, end)        
        f.seek(2 + 4 + 4 + dim * 4 * s, 0)
        features = np.fromfile(f, '<f', (e-s) * dim)
        features.resize(e-s, dim)
        
    if start != s or end != e:  # repeat first or/and last frame as required
        features = np.r_[np.repeat(features[[0]], s-start, axis=0), features, np.repeat(features[[-1]], end-e, axis=0)]
        
    return features


@check_path_existance
def write_spro4(features, outputFileName):
    """Write a feature stream in SPRO4 format.
    
    :param features: sequence of features to write
    :param outputFileName: name of the file to write to
    """
    nframes, dim = np.shape(features)  # get feature stream's dimensions
    f = open(outputFileName, 'wb')  # open outputFile
    f.write(struct.pack("H", dim))  # write feature dimension
    f.write(struct.pack("4b", 25, 0, 0, 0))  # write flag (not important)
    f.write(struct.pack("f", 100.0))  # write frequency of feature extraciton
    data = features.flatten()  # Write the data
    f.write(struct.pack('f' * len(data), *data))
    f.close()


@check_path_existance
def write_htk(features,
              outputFileName,
              fs=100,
              dt=9):
    """ Write htk feature file

            0. WAVEFORM Acoustic waveform
            1.  LPC Linear prediction coefficients
            2.  LPREFC LPC Reflection coefficients: -lpcar2rf([1 LPC]);LPREFC(1)=[];
            3.  LPCEPSTRA    LPC Cepstral coefficients
            4. LPDELCEP     LPC cepstral+delta coefficients (obsolete)
            5.  IREFC        LPC Reflection coefficients (16 bit fixed point)
            6.  MFCC         Mel frequency cepstral coefficients
            7.  FBANK        Log Fliter bank energies
            8.  MELSPEC      linear Mel-scaled spectrum
            9.  USER         User defined features
            10.  DISCRETE     Vector quantised codebook
            11.  PLP          Perceptual Linear prediction    
    
    :param features: vector for waveforms, one row per frame for other types
    :param outputFileName: name of the file to write to
    :param fs: feature sample in Hz
    :param dt: data type (also includes Voicebox code for generating data)
        
            0. WAVEFORM Acoustic waveform
            1.  LPC Linear prediction coefficients
            2.  LPREFC LPC Reflection coefficients: -lpcar2rf([1 LPC]);LPREFC(1)=[];
            3.  LPCEPSTRA    LPC Cepstral coefficients
            4. LPDELCEP     LPC cepstral+delta coefficients (obsolete)
            5.  IREFC        LPC Reflection coefficients (16 bit fixed point)
            6.  MFCC         Mel frequency cepstral coefficients
            7.  FBANK        Log Fliter bank energies
            8.  MELSPEC      linear Mel-scaled spectrum
            9.  USER         User defined features
            10.  DISCRETE     Vector quantised codebook
            11.  PLP          Perceptual Linear prediction
            12.  ANON
    """
    sampPeriod = 1./fs    
    
    pk = dt & 0x3f
    dt &= ~_K  # clear unsupported CRC bit
    features = np.atleast_2d(features)
    if pk == 0:
        features = features.reshape(-1, 1)
    with open(outputFileName, 'wb') as fh:
        fh.write(struct.pack(">IIHH", len(features)+(4 if dt & _C else 0), sampPeriod*1e7,
                             features.shape[1] * (2 if (pk in parms16bit or dt & _C) else 4), dt))
        if pk == 5:
            features *= 32767.0
        if pk in parms16bit:
            features = m.astype('>h')
        elif dt & _C:
            mmax, mmin = features.max(axis=0), features.min(axis=0)
            mmax[mmax == mmin] += 32767
            mmin[mmax == mmin] -= 32767  # to avoid division by zero for constant coefficients
            scale = 2 * 32767. / (mmax - mmin)
            bias = 0.5 * scale * (mmax + mmin)
            features = features * scale - bias
            scale.astype('>f').tofile(fh)
            bias.astype('>f').tofile(fh)
            features = m.astype('>h')
        else:
            features = features.astype('>f')
        features.tofile(fh)


def read_htk(inputFileName,
             labelFileName="",
             selectedLabel="",
             framePerSecond=100):
    """Read a sequence of features in HTK format

    :param inputFileName: name of the file to read from
    :param labelFileName: name of the label file to read from
    :param selectedLabel: label to select
    :param framePerSecond: number of frames per second
    
    :return: a tupple (d, fp, dt, tc, t) described below
    
    .. note::
    
        - d = data: column vector for waveforms, 1 row per frame for other types
        - fp = frame period in seconds
        - dt = data type (also includes Voicebox code for generating data)
        
            0. WAVEFORM Acoustic waveform
            1.  LPC Linear prediction coefficients
            2.  LPREFC LPC Reflection coefficients: -lpcar2rf([1 LPC]);LPREFC(1)=[];
            3.  LPCEPSTRA    LPC Cepstral coefficients
            4. LPDELCEP     LPC cepstral+delta coefficients (obsolete)
            5.  IREFC        LPC Reflection coefficients (16 bit fixed point)
            6.  MFCC         Mel frequency cepstral coefficients
            7.  FBANK        Log Fliter bank energies
            8.  MELSPEC      linear Mel-scaled spectrum
            9.  USER         User defined features
            10.  DISCRETE     Vector quantised codebook
            11.  PLP          Perceptual Linear prediction
            12.  ANON
            
        - tc = full type code = dt plus (optionally) 
                one or more of the following modifiers
                
            - 64  _E  Includes energy terms
            - 128  _N  Suppress absolute energy
            - 256  _D  Include delta coefs
            - 512  _A  Include acceleration coefs
            - 1024  _C  Compressed
            - 2048  _Z  Zero mean static coefs
            - 4096  _K  CRC checksum (not implemented yet)
            - 8192  _0  Include 0'th cepstral coef
            - 16384  _V  Attach VQ index
            - 32768  _T  Attach delta-delta-delta index
            
        - t = text version of type code e.g. LPC_C_K

    This function is a translation of the Matlab code from
    VOICEBOX is a MATLAB toolbox for speech processing.
    by  Mike Brookes
    Home page: `VOICEBOX <http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html>`
    """
    kinds = ['WAVEFORM', 'LPC', 'LPREFC', 'LPCEPSTRA', 'LPDELCEP', 'IREFC',
             'MFCC', 'FBANK', 'MELSPEC', 'USER', 'DISCRETE', 'PLP', 'ANON',
             '???']
    with open(inputFileName, 'rb') as fid:
        nf = struct.unpack(">l", fid.read(4))[0]  # number of frames
        # frame interval (in seconds)
        fp = struct.unpack(">l", fid.read(4))[0] * 1.e-7
        by = struct.unpack(">h", fid.read(2))[0]  # bytes per frame
        tc = struct.unpack(">h", fid.read(2))[0]  # type code
        tc += 65536 * (tc < 0)
        cc = 'ENDACZK0VT'  # list of suffix codes
        nhb = len(cc)  # nbumber of suffix codes
        ndt = 6  # number of bits for base type
        hb = list(int(math.floor(tc * 2 ** x))
                  for x in range(- (ndt + nhb), -ndt + 1))
        # extract bits from type code
        hd = list(hb[x] - 2 * hb[x - 1] for x in range(nhb, 0, -1))
        # low six bits of tc represent data type
        dt = tc - hb[-1] * 2 ** ndt

        # hd(7)=1 CRC check
        # hd(5)=1 compressed data
        if dt == 5:
            fid.seek(0, 2)  # Go to te end of the file
            flen = fid.tell()  # get the position
            fid.seek(0, 0)  # go back to the begining of the file
            if flen > 14 + by * nf:  # if file too long
                dt = 2  # change type to LPRFEC
                hd[5] = 1  # set compressed flag
                nf += 4  # frame count doesn't include
                # compression constants in this case

        # 16 bit data for waveforms, IREFC and DISCRETE
        if any([dt == x for x in [0, 5, 10]]):
            ndim = int(by * nf / 2)
            data = np.asarray(struct.unpack(">" + "h" *
                                            ndim, fid.read(2 * ndim)))
            d = data.reshape(nf, by / 2)
            if dt == 5:
                d /= 32767  # scale IREFC
        else:
            if hd[5]:  # compressed data - first read scales
                nf -= 4  # frame count includes compression constants
                ncol = int(by / 2)
                scales = np.asarray(struct.unpack(">" +
                                                  "f" * ncol,
                                                  fid.read(4 * ncol)))
                biases = np.asarray(struct.unpack(">" + "f" * ncol, fid.read(4 * ncol)))
                data = np.asarray(struct.unpack(">" + "h" * ncol * nf, fid.read(2 * ncol * nf)))
                d = data.reshape(nf, ncol)
                d = d + biases
                d = d / scales
            else:
                data = np.asarray(struct.unpack(">" + "f" * int(by / 4) * nf,
                                                fid.read(by * nf)))
                d = data.reshape(nf, by / 4)

    t = kinds[min(dt, len(kinds) - 1)]

    lbl = np.ones(np.shape(d)[0]).astype(bool)
    if not labelFileName == "":
        lbl = read_label(labelFileName, selectedLabel, framePerSecond)

    d = d[lbl, :]

    return d, fp, dt, tc, t


def read_htk_segment(inputFileName,
                     start=0,
                     stop=None):
    """Read a segment from a stream in SPRO4 format. Return the features in the
    range start:end
    In case the start and end cannot be reached, the first or last feature are copied
    so that the length of the returned segment is always end-start
    
    :param inputFileName: name of the feature file to read from or file-like 
        object alowing to seek in the file
    :param start: index of the first frame to read (start at zero)
    :param stop: index of the last frame following the segment to read.
       end < 0 means that end is the value of the right_context to add 
       at the end of the file
       
    :return: a sequence of features in a ndarray of length end-start
    """
    try:
        fh = open(inputFileName, 'rb')
    except TypeError:
        fh = inputFileName
    try:
        fh.seek(0)
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">IIHH", fh.read(12))
        pk = parmKind & 0x3f
        if parmKind & _C:
            scale, bias = np.fromfile(fh, '>f', sampSize).reshape(2, sampSize/2)
            nSamples -= 4
        s, e = max(0, start), min(nSamples, stop)
        fh.seek(s*sampSize, 1)
        dtype, _bytes = ('>h', 2) if parmKind & _C or pk in parms16bit else ('>f', 4)
        m = np.fromfile(fh, dtype, (e - s) * sampSize / _bytes).reshape(e - s, sampSize / _bytes)
        if parmKind & _C:
            m = (m + bias) / scale
        if pk == IREFC:
            m /= 32767.0
        if pk == WAVEFORM:
            m = m.ravel()
    finally:
        if fh is not inputFileName:
            fh.close()
    if start != s or stop != e:  # repeat first or/and last frame as required
        m = np.r_[np.repeat(m[[0]], s-start, axis=0), m, np.repeat(m[[-1]], stop-e, axis=0)]
    return m


def read_feature_segment(inputFileName,
                         file_format='spro4',
                         start=0,
                         stop=None):
    if file_format == 'spro4':
        m = read_spro4_segment(inputFileName, start, stop)
        return m
    elif file_format == 'htk':
        m = read_htk_segment(inputFileName, start, stop)
        return m
    else:
        print("Error: unsupported feature file format")
