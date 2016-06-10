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

:mod:`sidekit_io` provides methods to read and write from and to different 
formats.
"""
import os
import struct
import array
import numpy
import pickle
import gzip
import logging
from .sidekit_wrappers import check_path_existance
try:
    import h5py
    h5py_loaded = True
except ImportError:
    h5py_loaded = False


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
    return numpy.array(data)


def read_matrix(filename):
    """Read matrix in ALIZE binary format and return a ndarray
    
    :param filename: name of the file to read from
    
    :return: a numpy.ndarray object
    """
    with open(filename, 'rb') as f:
        mDim = struct.unpack("<2l", f.read(8))
        data = array.array("d")
        data.fromstring(f.read())
        T = numpy.array(data)
        T.resize(mDim[0], mDim[1])
    return T


@check_path_existance
def write_matrix(m, filename):
    """Write a  matrix in ALIZE binary format

    :param m: a 2-dimensional ndarray
    :param filename: name of the file to write in

    :exception: TypeError if m is not a 2-dimensional ndarray
    """
    if not m.ndim == 2:
        raise TypeError("To write vector, use write_vect")
    else:
        with open(filename, 'wb') as mf:
            data = numpy.array(m.flatten())
            mf.write(struct.pack("<l", m.shape[0]))
            mf.write(struct.pack("<l", m.shape[1]))
            mf.write(struct.pack("<" + "d" * m.shape[0] * m.shape[1], *data))


@check_path_existance
def write_vect(v, filename):
    """Write a  vector in ALIZE binary format

    :param v: a 1-dimensional ndarray
    :param filename: name of the file to write in
    
    :exception: TypeError if v is not a 1-dimensional ndarray
    """
    if not v.ndim == 1:
        raise TypeError("To write matrix, use write_matrix")
    else:
        with open(filename, 'wb') as mf:
            mf.write(struct.pack("<l", 1))
            mf.write(struct.pack("<l", v.shape[0]))
            mf.write(struct.pack("<" + "d" * v.shape[0], *v))


@check_path_existance
def write_matrix_int(m, filename):
    """Write matrix of int in ALIZE binary format
    
    :param m: a 2-dimensional ndarray of int
    :param filename: name of the file to write in
    """
    if not m.ndim == 2:
        raise TypeError("To write vector, use write_vect")
    if not m.dtype == 'int64':
        raise TypeError("m must be a ndarray of int64")
    with open(filename, 'wb') as mf:
        data = numpy.array(m.flatten())
        mf.write(struct.pack("<l", m.shape[0]))
        mf.write(struct.pack("<l", m.shape[1]))
        mf.write(struct.pack("<" + "l" * m.shape[0] * m.shape[1], *data))


def read_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


@check_path_existance
def write_pickle(obj, filename):
    if not (os.path.exists(os.path.dirname(filename)) or os.path.dirname(filename) == ''):
        os.makedirs(os.path.dirname(filename))
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)


@check_path_existance
def write_tv_hdf5(data, output_filename):
    tv = data[0]
    tv_mean = data[1]
    tv_sigma = data[2]
    d = dict()
    d['tv/tv'] = tv
    d['tv/tv_mean'] = tv_mean
    d['tv/tv_sigma'] = tv_sigma
    write_dict_hdf5(d, output_filename)


def read_tv_hdf5(input_filename):
    with h5py.File(input_filename, "r") as f:
        tv = f.get("tv/tv").value
        tv_mean = f.get("tv/tv_mean").value
        tv_sigma = f.get("tv/tv_sigma").value
    return tv, tv_mean, tv_sigma


@check_path_existance
def write_dict_hdf5(data, output_filename):
    with h5py.File(output_filename, "w") as f:
        for key in data:
            value = data[key]
            if isinstance(value, numpy.ndarray) or isinstance(value, list):
                f.create_dataset(key,
                                 data=value,
                                 compression="gzip",
                                 fletcher32=True)
            else:
                f.create_dataset(key, data=value)


def read_dict_hdf5(input_filename):
    data = dict()
    with h5py.File(input_filename, "r") as f:
        for key in f.keys():
            logging.debug('key: '+key)
            for key2 in f.get(key).keys():
                data[key+'/'+key2] = f.get(key).get(key2).value
    return data


@check_path_existance
def write_norm_hdf5(data, output_filename):
    with h5py.File(output_filename, "w") as f:
        means = data[0]
        covs = data[1]
        f.create_dataset("norm/means", data=means,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset("norm/covs", data=covs,
                         compression="gzip",
                         fletcher32=True)


def read_norm_hdf5(statserver_filename):
    with h5py.File(statserver_filename, "r") as f:
        means = f.get("norm/means").value
        covs = f.get("norm/covs").value
    return means, covs


@check_path_existance
def write_plda_hdf5(data, output_filename):
    mean = data[0]
    mat_f = data[1]
    mat_g = data[2]
    sigma = data[3]
    with h5py.File(output_filename, "w") as f:
        f.create_dataset("plda/mean", data=mean,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset("plda/f", data=mat_f,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset("plda/g", data=mat_g,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset("plda/sigma", data=sigma,
                         compression="gzip",
                         fletcher32=True)


def read_plda_hdf5(statserver_filename):
    with h5py.File(statserver_filename, "r") as f:
        mean = f.get("plda/mean").value
        mat_f = f.get("plda/f").value
        mat_g = f.get("plda/g").value
        sigma = f.get("plda/sigma").value
    return mean, mat_f, mat_g, sigma


@check_path_existance
def write_fa_hdf5(data, output_filename):
    mean = data[0]
    f = data[1]
    g = data[2]
    h = data[3]
    sigma = data[4]
    with h5py.File(output_filename, "w") as fh:
        kind = numpy.zeros(5, dtype="int16") # FA with 5 matrix
        if mean is not None:
            kind[0] = 1
            fh.create_dataset("fa/mean", data=mean,
                             compression="gzip",
                             fletcher32=True)
        if f is not None:
            kind[1] = 1
            fh.create_dataset("fa/f", data=f,
                             compression="gzip",
                             fletcher32=True)
        if g is not None:
            kind[2] = 1
            fh.create_dataset("fa/g", data=g,
                             compression="gzip",
                             fletcher32=True)
        if h is not None:
            kind[3] = 1
            fh.create_dataset("fa/h", data=h,
                             compression="gzip",
                             fletcher32=True)
        if sigma is not None:
            kind[4] = 1
            fh.create_dataset("fa/sigma", data=sigma,
                             compression="gzip",
                             fletcher32=True)
        fh.create_dataset("fa/kind", data=kind,
                         compression="gzip",
                         fletcher32=True)


def read_fa_hdf5(statserver_filename):
    with h5py.File(statserver_filename, "r") as fh:
        kind = fh.get("fa/kind").value
        mean = f = g = h = sigma = None
        if kind[0] != 0:
            mean = fh.get("fa/mean").value
        if kind[1] != 0:
            f = fh.get("fa/f").value
        if kind[2] != 0:
            g = fh.get("fa/g").value
        if kind[3] != 0:
            h = fh.get("fa/h").value
        if kind[4] != 0:
            sigma = fh.get("fa/sigma").value
    return mean, f, g, h, sigma


def h5merge(output_filename, input_filename_list):
    with h5py.File(output_filename, "w") as fo:
        for ifn in input_filename_list:
            logging.debug('read '+ifn)
            data = read_dict_hdf5(ifn)
            for key in data:
                value = data[key]
                if isinstance(value, numpy.ndarray) or isinstance(value, list):
                    fo.create_dataset(key,
                                      data=value,
                                      compression="gzip",
                                      fletcher32=True)
                else:
                    fo.create_dataset(key, data=value)


def init_logging(level=logging.INFO, filename=None):
    numpy.set_printoptions(linewidth=250, precision=4)
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

