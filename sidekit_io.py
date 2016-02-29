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

:mod:`sidekit_io` provides methods to read and write from and to different 
formats.
"""
import os
import struct
import array
import numpy as np
import pickle
import gzip
import logging
from sidekit.sidekit_wrappers import *


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def read_vect(filename):
    """Read vector in ALIZE binary format and return an array
    
    :param filename: name of the file to read from
    
    :return: a numpy.ndarray object
    """
    with open(filename, 'rb') as f:
        struct.unpack("<2l", f.read(8))
        data = array.array("d")
        data.fromstring(f.read())
    return np.array(data)


def read_matrix(filename):
    """Read matrix in ALIZE binary format and return a ndarray
    
    :param filename: name of the file to read from
    
    :return: a numpy.ndarray object
    """
    with open(filename, 'rb') as f:
        mDim = struct.unpack("<2l", f.read(8))
        data = array.array("d")
        data.fromstring(f.read())
        T = np.array(data)
        T.resize(mDim[0], mDim[1])
    return T


@check_path_existance
def write_matrix(M, filename):
    """Write a  matrix in ALIZE binary format

    :param M: a 2-dimensional ndarray 
    :param filename: name of the file to write in

    :exception: TypeError if M is not a 2-dimensional ndarray
    """
    if not M.ndim == 2:
        raise TypeError("To write vector, use write_vect")
    else:
        with open(filename, 'wb') as mf:
            data = np.array(M.flatten())
            mf.write(struct.pack("<l", M.shape[0]))
            mf.write(struct.pack("<l", M.shape[1]))
            mf.write(struct.pack("<" + "d" * M.shape[0] * M.shape[1], *data))


@check_path_existance
def write_vect(V, filename):
    """Write a  vector in ALIZE binary format

    :param V: a 1-dimensional ndarray 
    :param filename: name of the file to write in
    
    :exception: TypeError if V is not a 1-dimensional ndarray
    """
    if not V.ndim == 1:
        raise TypeError("To write matrix, use write_matrix")
    else:
        with open(filename, 'wb') as mf:
            mf.write(struct.pack("<l", 1))
            mf.write(struct.pack("<l", V.shape[0]))
            mf.write(struct.pack("<" + "d" * V.shape[0], *V))


@check_path_existance
def write_matrix_int(M, filename):
    """Write matrix of int in ALIZE binary format
    
    :param M: a 2-dimensional ndarray of int
    :param filename: name of the file to write in
    """
    if not M.ndim == 2:
        raise TypeError("To write vector, use write_vect")
    if not M.dtype == 'int64':
        raise TypeError("M must be a ndarray of int64")
    with open(filename, 'wb') as mf:
        data = np.array(M.flatten())
        mf.write(struct.pack("<l", M.shape[0]))
        mf.write(struct.pack("<l", M.shape[1]))
        mf.write(struct.pack("<" + "l" * M.shape[0] * M.shape[1], *data))


def read_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


@check_path_existance
def write_pickle(obj, filename):
    if not (os.path.exists(os.path.dirname(filename)) or os.path.dirname(filename) == ''):
        os.makedirs(os.path.dirname(filename))
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)


def init_logging(level=logging.INFO, filename=None):
    np.set_printoptions(linewidth=250, precision=4)
    frm = '%(asctime)s - %(levelname)s - %(message)s'

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(format=frm, level=level)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setFormatter(logging.Formatter(frm))
        fh.setLevel(level)
        root.addHandler(fh)

