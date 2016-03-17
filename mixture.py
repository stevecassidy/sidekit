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

:mod:`mixture` provides methods to manage Gaussian mixture models

"""
import numpy as np
import struct
import copy
import ctypes
import sys
import multiprocessing
import pickle
import gzip
import logging
import os
import warnings

from sidekit.sidekit_wrappers import *

if sys.version_info.major == 3:
    import queue as Queue
else:
    import Queue

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


def sum_log_probabilities(lp):
    """Sum log probabilities in a secure manner to avoid extreme values

    :param lp: ndarray of log-probabilities to sum
    """
    pp_max = np.max(lp, axis=1)
    log_lk = pp_max + np.log(np.sum(np.exp((lp.transpose() - pp_max).T), axis=1))
    ind = ~np.isfinite(pp_max)
    if sum(ind) != 0:
        log_lk[ind] = pp_max[ind]
    pp = np.exp((lp.transpose() - log_lk).transpose())
    llk = log_lk.sum()
    return pp, llk


class Mixture(object):
    """
    A class for Gaussian Mixture Model storage.
    For more details about Gaussian Mixture Models (GMM) you can refer to
    [Bimbot04]_.

    So far, only diagonal covariance Gaussian distributions are available.
    A full covariance version of the GMMs might be implemented in the future.
    
    :attr w: array of weight parameters
    :attr mu: ndarray of mean parameters, each line is one distribution 
    :attr invcov: ndarray of inverse co-variance parameters, 2-dimensional 
        for diagonal co-variance distribution 3-dimensional for full co-variance
    :attr invchol: 3-dimensional ndarray containing lower cholesky 
        decomposition of the inverse co-variance matrices
    :attr cst: array of constant computed for each distribution
    :attr det: array of determinant for each distribution
    
    """

    def __init__(self, mixtureFileName='', mixtureFileFormat='hdf5',
                 name='empty'):
        """Initialize a Mixture from a file or as an empty Mixture.
        
        :param mixtureFileName: name of the file to read from, if empty, initialize 
            an empty mixture
        :param mixtureFileFormat: format of the Mixture file to read from. Can be:
            - alize
            - hdf5 (default)
            - htk
            - pickle
        """
        self.w = np.array([])
        self.mu = np.array([])
        self.invcov = np.array([])
        self.cov_var_ctl = np.array([])
        self.cst = np.array([])
        self.det = np.array([])
        self.name = name
        self.A = 0
            
        if mixtureFileName == '':
            pass
        elif mixtureFileFormat.lower() == 'pickle':
            self.read_pickle(mixtureFileName)
        elif mixtureFileFormat.lower() in ['hdf5', 'h5']:
            if h5py_loaded:
                self.read_hdf5(mixtureFileName)
            else:
                raise Exception('H5PY is not installed, chose another' + ' format to load your Mixture')
        elif mixtureFileFormat.lower() == 'alize':
            self.read_alize(mixtureFileName)
        elif mixtureFileFormat.lower() == 'htk':
            self.read_htk(mixtureFileName)
        else:
            raise Exception("Wrong mixtureFileFormat")
    
    @accepts('Mixture', 'Mixture', debug=2)
    def __add__(self, other):
        """Overide the sum for a mixture.
        Weight, means and inv_covariances are added, det and cst are
        set to 0
        """
        new_mixture = Mixture()
        new_mixture.w = self.w + other.w
        new_mixture.mu = self.mu + other.mu
        new_mixture.invcov = self.invcov + other.invcov
        return new_mixture

    def _serialize(self):
        """
        Serialization is necessary to share the memomry when running multiprocesses
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            sh = self.w.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.w.size)
            self.w = np.ctypeslib.as_array(tmp.get_obj())
            self.w = self.w.reshape(sh)

            sh = self.mu.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.mu.size)
            self.mu = np.ctypeslib.as_array(tmp.get_obj())
            self.mu = self.mu.reshape(sh)

            sh = self.invcov.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.invcov.size)
            self.invcov = np.ctypeslib.as_array(tmp.get_obj())
            self.invcov = self.invcov.reshape(sh)

            sh = self.cov_var_ctl.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.cov_var_ctl.size)
            self.cov_var_ctl = np.ctypeslib.as_array(tmp.get_obj())
            self.cov_var_ctl = self.cov_var_ctl.reshape(sh)

            sh = self.cst.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.cst.size)
            self.cst = np.ctypeslib.as_array(tmp.get_obj())
            self.cst = self.cst.reshape(sh)

            sh = self.det.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.det.size)
            self.det = np.ctypeslib.as_array(tmp.get_obj())
            self.det = self.det.reshape(sh)

    def read(self, inputFileName):
        """Read information from a file and constructs a Mixture object. The
        type of file is deduced from the extension. The extension must be
        '.hdf5' or '.h5' for a HDF5 file and '.p' for pickle, '.gmm' for ALIZE
        and '.htk' for HTK.
        In order to use different extension, use specific functions.

    :param inputFileName: name of the file o read from
    """
        extension = os.path.splitext(inputFileName)[1][1:].lower()
        if extension == 'p':
            self.read_pickle(inputFileName)
        elif extension in ['hdf5', 'h5']:
            if h5py_loaded:
                self.read_hdf5(inputFileName)
            else:
                raise Exception('H5PY is not installed, chose another' + ' format to load your Scores')
        elif extension == 'gmm':
            self.read_alize(inputFileName)
        elif extension == 'htk':
            self.read_htk(inputFileName)
        else:
            raise Exception('Error: unknown extension')
            
    def read_hdf5(self, mixtureFileName):
        """Read a Mixture in hdf5 format

        :param mixtureFileName: name of the file to read from
        """
        with h5py.File(mixtureFileName, 'r') as f:
            self.w = f.get('/w').value
            self.w.resize(np.max(self.w.shape))
            self.mu = f.get('/mu').value
            self.invcov = f.get('/invcov').value
            self.cov_var_ctl = f.get('/cov_var_ctl').value
            self.cst = f.get('/cst').value
            self.det = f.get('/det').value
            self.A = f.get('/A').value

    def read_pickle(self, inputFileName):
        """Read IdMap in PICKLE format.
        
        :param inputFileName: name of the file to read from
        """
        with gzip.open(inputFileName, 'rb') as f:
            gmm = pickle.load(f)
            self.w = gmm.w
            self.mu = gmm.mu
            self.invcov = gmm.invcov
            self.cst = gmm.cst
            self.det = gmm.det
        self._compute_all()

    def read_alize(self, mixtureFileName):
        """Read a Mixture in alize raw format

        :param mixtureFileName: name of the file to read from
        """
        logging.info('Reading %s', mixtureFileName)
        with open(mixtureFileName, 'rb') as f:
            distrib_nb = struct.unpack("I", f.read(4))[0]
            vect_size = struct.unpack("<I", f.read(4))[0]

            # resize all attributes
            self.w = np.zeros(distrib_nb, "d")
            self.invcov = np.zeros((distrib_nb, vect_size), "d")
            self.mu = np.zeros((distrib_nb, vect_size), "d")
            self.cst = np.zeros(distrib_nb, "d")
            self.det = np.zeros(distrib_nb, "d")

            for d in range(distrib_nb):
                self.w[d] = struct.unpack("<d", f.read(8))[0]
            for d in range(distrib_nb):
                self.cst[d] = struct.unpack("d", f.read(8))[0]
                self.det[d] = struct.unpack("d", f.read(8))[0]
                f.read(1)
                for c in range(vect_size):
                    self.invcov[d, c] = struct.unpack("d", f.read(8))[0]
                for c in range(vect_size):
                    self.mu[d, c] = struct.unpack("d", f.read(8))[0]
        self._compute_all()

    def read_htk(self, mixtureFileName, beginHmm=False, state2=False):
        """Read a Mixture in HTK format
        
        :param mixtureFileName: name of the file to read from
        :param beginHmm: boolean
        :param state2: boolean
        """
        with open(mixtureFileName, 'rb') as f:
            lines = [line.rstrip() for line in f]

        distrib = 0
        vect_size = 0
        for i in range(len(lines)):

            if lines[i] == '':
                break

            w = lines[i].split()

            if w[0] == '<NUMMIXES>':
                distrib_nb = int(w[1])
                self.w.resize(distrib_nb)
                self.cst.resize(distrib_nb)
                self.det.resize(distrib_nb)

            if w[0] == '<BEGINHMM>':
                beginHmm = True

            if w[0] == '<STATE>':
                state2 = True

            if beginHmm & state2:

                if w[0].upper() == '<MIXTURE>':
                    distrib = int(w[1]) - 1
                    self.w[distrib] = np.double(w[2])

                elif w[0].upper() == '<MEAN>':
                    if vect_size == 0:
                        vect_size = int(w[1])
                    self.mu.resize(distrib_nb, vect_size)
                    i += 1
                    self.mu[distrib, :] = np.double(lines[i].split())

                elif w[0].upper() == '<VARIANCE>':
                    if self.invcov.shape[0] == 0:
                        vect_size = int(w[1])
                    self.invcov.resize(distrib_nb, vect_size)
                    i += 1
                    C = np.double(lines[i].split())
                    self.invcov[distrib, :] = 1 / C

                elif w[0].upper() == '<INVCOVAR>':
                    raise Exception("we don't manage full covariance model")
                elif w[0].upper() == '<GCONST>':
                    self.cst[distrib] = np.exp(-.05 * np.double(w[1]))
        self._compute_all()

    @check_path_existance
    def save(self, outputFileName):
        """Save the Mixture object to file. The format of the file 
        to create is set accordingly to the extension of the filename.
        This extension can be '.p' for pickle format, '.hdf5' and '.h5' 
        for HDF5 format, '.gmm' for ALIZE format (HTK not implemented yet)

        :param outputFileName: name of the file to write to
        """
        extension = os.path.splitext(outputFileName)[1][1:].lower()
        if extension == 'p':
            self.save_pickle(outputFileName)
        elif extension in ['hdf5', 'h5']:
            if h5py_loaded:
                self.save_hdf5(outputFileName)
            else:
                raise Exception('h5py is not installed, chose another' + ' format to load your IdMap')
        elif extension == 'gmm':
            self.save_alize(outputFileName)
        else:
            raise Exception('Wrong output format, must be pickle or hdf5')

    @check_path_existance
    def save_alize(self, mixtureFileName):
        """Save a mixture in alize raw format

        :param mixtureFileName: name of the file to write in     
        """
        with open(mixtureFileName, 'wb') as of:
            # write the number of distributions per state
            of.write(struct.pack("<I", self.distrib_nb()))
            # Write the dimension of the features
            of.write(struct.pack("<I", self.dim()))
            # Weights
            of.write(struct.pack("<" + "d" * self.w.shape[0], *self.w))
            # For each distribution
            for d in range(self.distrib_nb()):
                # Write the constant
                of.write(struct.pack("<d", self.cst[d]))
                # Write the determinant
                of.write(struct.pack("<d", self.det[d]))
                # write a meaningless char for compatibility purpose
                of.write(struct.pack("<c", bytes(1)))
                # Covariance
                of.write(
                    struct.pack("<" + "d" * self.dim(), *self.invcov[d, :]))
                # Means
                of.write(struct.pack("<" + "d" * self.dim(), *self.mu[d, :]))

    @check_path_existance
    def save_hdf5(self, mixtureFileName):
        """Save a Mixture in hdf5 format

        :param mixtureFileName: the name of the file to write in
        """
        f = h5py.File(mixtureFileName, 'w')
        f.create_dataset('/w', self.w.shape, "d", self.w,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset('/mu', self.mu.shape, "d", self.mu,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset('/invcov', self.invcov.shape, "d", self.invcov,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset('/cov_var_ctl', self.cov_var_ctl.shape, "d", 
                         self.cov_var_ctl,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset('/cst', self.cst.shape, "d", self.cst,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset('/det', self.det.shape, "d", self.det,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset('/A', self.A.shape, "d", self.A,
                         compression="gzip",
                         fletcher32=True)
        
        f.close()

    @check_path_existance
    def save_pickle(self, outputFileName):
        """Save Ndx in PICKLE format. Convert all data into float32 
        before saving, note that the conversion doesn't apply in Python 2.X
        
        :param outputFileName: name of the file to write to
        """
        with gzip.open(outputFileName, 'wb') as f:
            self.w.astype('float32', copy=False)
            self.mu.astype('float32', copy=False)
            self.invcov.astype('float32', copy=False)
            self.cov_var_ctl.astype('float32', copy=False)
            self.cst.astype('float32', copy=False)
            self.det.astype('float32', copy=False)
            pickle.dump(self, f)

    @check_path_existance
    def save_htk(self, mixtureFileName):
        """Save a Mixture in HTK format
        
        :param mixtureFileName: the name of the file to write in
        """
        # TODO
        pass

    def distrib_nb(self):
        """Return the number of distribution of the Mixture
        
        :return: the number of distribution in the Mixture
        """
        return self.w.shape[0]

    def dim(self):
        """Return the dimension of distributions of the Mixture
        
        :return: an integer, size of the acoustic vectors
        """
        return self.mu.shape[1]

    def sv_size(self):
        """Return the dimension of the super-vector
        
        :return: an integer, size of the mean super-vector
        """
        return self.mu.shape[1] * self.w.shape[0]

    def _compute_all(self):
        """Compute determinant and constant values for each distribution"""
        self.det = 1.0 / np.prod(self.invcov, axis=1)
        self.cst = 1.0 / (np.sqrt(self.det) *
                          (2.0 * np.pi) ** (self.dim() / 2.0))
        self.A = (np.square(self.mu) * self.invcov).sum(1) - 2.0 * (np.log(self.w) + np.log(self.cst))

    def validate(self):
        """Verify the format of the Mixture
        
        :return: a boolean giving the status of the Mixture
        """
        cov = 'diag'
        ok = (self.w.ndim == 1)
        ok &= (self.det.ndim == 1)
        ok &= (self.cst.ndim == 1)
        ok &= (self.mu.ndim == 2)
        if self.invcov.ndim == 3:
            cov = 'full'
        else:
            ok &= (self.invcov.ndim == 2)

        ok &= (self.w.shape[0] == self.mu.shape[0])
        ok &= (self.w.shape[0] == self.cst.shape[0])
        ok &= (self.w.shape[0] == self.det.shape[0])
        if cov == 'diag':
            ok &= (self.invcov.shape == self.mu.shape)
        else:
            ok &= (self.w.shape[0] == self.invcov.shape[0])
            ok &= (self.mu.shape[1] == self.invcov.shape[1])
            ok &= (self.mu.shape[1] == self.invcov.shape[2])
        return ok

    def get_mean_super_vector(self):
        """Return mean super-vector
        
        :return: an array, super-vector of the mean coefficients
        """
        sv = self.mu.flatten()
        return sv

    def get_invcov_super_vector(self):
        """Return Inverse covariance super-vector
        
        :return: an array, super-vector of the inverse co-variance coefficients
        """
        assert self.invcov.ndim == 2, 'Must be diagonal co-variance.'
        sv = self.invcov.flatten()
        return sv

    def compute_log_posterior_probabilities(self, cep, mu=None):
        """ Compute log posterior probabilities for a set of feature frames.
        
        :param cep: a set of feature frames in a ndarray, one feature per row
        :param mu: a mean super-vector to replace the ubm's one. If it is an empty 
              vector, use the UBM
        
        :return: A ndarray of log-posterior probabilities corresponding to the 
              input feature set.
        """            
        if cep.ndim == 1:
            cep = cep[:, np.newaxis]
        A = self.A
        if mu is None:
            mu = self.mu
        else:
            # for MAP, Compute the data independent term
            A = (np.square(mu.reshape(self.mu.shape)) * self.invcov).sum(1) \
               - 2.0 * (np.log(self.w) + np.log(self.cst))

        # Compute the data independent term
        B = np.dot(np.square(cep), self.invcov.T) \
            - 2.0 * np.dot(cep, np.transpose(mu.reshape(self.mu.shape) * self.invcov))
        
        # Compute the exponential term
        lp = -0.5 * (B + A)
        return lp

    def varianceControl(self, cov, flooring, ceiling, cov_ctl):
        """varianceControl for Mixture (florring and ceiling)

        :param cov: covariance to control
        :param flooring: float, florring value
        :param ceiling: float, ceiling value
        :param cov_ctl: co-variance to consider for flooring and ceiling
        """
        floor = flooring * cov_ctl
        ceil = ceiling * cov_ctl

        to_floor = np.less_equal(cov, floor)
        to_ceil = np.greater_equal(cov, ceil)

        cov[to_floor] = floor[to_floor]
        cov[to_ceil] = ceil[to_ceil]
        return cov

    def _reset(self):
        """Set all the Mixture values to ZERO"""
        self.cst.fill(0.0)
        self.det.fill(0.0)
        self.w.fill(0.0)
        self.mu.fill(0.0)
        self.invcov.fill(0.0)
        self.A = 0.0

    def _split_ditribution(self):
        """Split each distribution into two depending on the principal
            axis of variance."""
        sigma = 1.0 / self.invcov
        sig_max = np.max(sigma, axis=1)
        arg_max = np.argmax(sigma, axis=1)

        shift = np.zeros(self.mu.shape)
        for x, y, z in zip(range(arg_max.shape[0]), arg_max, sig_max):
            shift[x, y] = np.sqrt(z)

        self.mu = np.vstack((self.mu - shift, self.mu + shift))
        self.invcov = np.vstack((self.invcov, self.invcov))
        self.w = np.concatenate([self.w, self.w]) * 0.5
        self.cst = np.zeros(self.w.shape)
        self.det = np.zeros(self.w.shape)
        self.cov_var_ctl = np.vstack((self.cov_var_ctl, self.cov_var_ctl))

        self._compute_all()

    def _expectation(self, accum, cep):
        """Expectation step of the EM algorithm. Calculate the expected value 
            of the log likelihood function, with respect to the conditional 
            distribution.
        
        :param accum: a Mixture object to store the accumulated statistics
        :param cep: a set of input feature frames
        
        :return loglk: float, the log-likelihood computed over the input set of 
              feature frames.
        """
        if cep.ndim == 1:
            cep = cep[:, np.newaxis]
        lp = self.compute_log_posterior_probabilities(cep)
        pp, loglk = sum_log_probabilities(lp)        

        # zero order statistics
        accum.w += pp.sum(0)
        # first order statistics
        accum.mu += np.dot(cep.T, pp).T
        # second order statistics
        accum.invcov += np.dot(np.square(cep.T), pp).T

        # return the log-likelihood
        return loglk

    @process_parallel_lists
    def _expectation_list(self, stat_acc, feature_list, feature_server, llk_acc=np.zeros(1),  numThread=1):
        """Expectation step of the EM algorithm. Calculate the expected value 
            of the log likelihood function, with respect to the conditional 
            distribution.
        
        :param accum: a Mixture object to store the accumulated statistics
        :param cep: a set of input feature frames
        
        :return loglk: float, the log-likelihood computed over the input set of 
              feature frames.
        """
        stat_acc._reset()
        feature_server.keep_all_features = False
        for feat in feature_list:
            cep = feature_server.load(feat)[0][0]
            llk_acc[0] += self._expectation(stat_acc, cep)

    def _maximization(self, accum, ceil_cov=10, floor_cov=1e-2):
        """Re-estimate the parmeters of the model which maximize the likelihood
            on the data.
        
        :param accum: a Mixture in which statistics computed during the E step 
              are stored
        :param floor_cov: a constant; minimum bound to consider, default is 1e-200
        """
        self.w = accum.w / np.sum(accum.w)
        self.mu = accum.mu / accum.w[:, np.newaxis]
        cov = accum.invcov / accum.w[:, np.newaxis] - np.square(self.mu)
        cov = self.varianceControl(cov, floor_cov, ceil_cov, self.cov_var_ctl)
        self.invcov = 1.0 / cov
        self._compute_all()

    def _init(self, cep):
        """Initialize a Mixture as a single Gaussian distribution which 
            mean and covariance are computed on a set of feature frames
        
        :param cep: a ndarray of feature frames to initialize the distribution,
              one feature per row
        """
        logging.debug('Mixture init: mu')
        self.mu = cep.mean(axis=0)[None]
        logging.debug('Mixture init: invcov')
        self.invcov = (cep.shape[0] /
                       np.sum(np.square(cep - self.mu), axis=0))[None]
        logging.debug('Mixture init: w')
        self.w = np.asarray([1.0])
        self.cst = np.zeros(self.w.shape)
        self.det = np.zeros(self.w.shape)
        self.cov_var_ctl = 1.0 / copy.deepcopy(self.invcov)
        self._compute_all()

    def EM_split(self, fs, featureList, distrib_nb,
                 iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8), numThread=1,
                 llk_gain=0.01):
        """Expectation-Maximization estimation of the Mixture parameters.
        
        :param fs: sidekit.FeaturesServer used to load data
        :param featureList: list of feature files to train the GMM
        :param distrib_nb: final number of distributions
        :param iterations: list of iteration number for each step of the learning process
        :param numThread: number of thread to launch for parallel computing
        :param llk_gain: limit of the training gain. Stop the training when gain between two iterations is less than this value
        
        :return llk: a list of log-likelihoods obtained after each iteration
        """
        llk = []
        logging.debug('EM Split init')
        self._init(fs.load(featureList[0])[0][0])

        # for N iterations:
        for it in iterations[:int(np.log2(distrib_nb))]:
            logging.debug('EM split it: %d', it)
            self._split_ditribution()

            # initialize the accumulator
            accum = copy.deepcopy(self)

            for i in range(it):
                accum._reset()

                # serialize the accum
                accum._serialize()
                llk_acc = np.zeros(1)
                sh = llk_acc.shape
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                    llk_acc = np.ctypeslib.as_array(tmp.get_obj())
                    llk_acc = llk_acc.reshape(sh)

                logging.debug('Expectation')
                # E step
                self._expectation_list(stat_acc=accum, 
                                       feature_list=featureList, 
                                       feature_server=fs,
                                       llk_acc=llk_acc, 
                                       numThread=numThread)
                llk.append(llk_acc[0] / np.sum(accum.w))

                # M step
                logging.debug('Maximisation')
                self._maximization(accum)
                if i > 0:
                    # gain = llk[-1] - llk[-2]
                    # if gain < llk_gain:
                        # logging.debug(
                        #    'EM (break) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        #    self.mu.shape[0], i + 1, it, gain, self.name,
                        #    len(cep))
                    #    break
                    # else:
                        # logging.debug(
                        #    'EM (continu) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        #    self.mu.shape[0], i + 1, it, gain, self.name,
                        #    len(cep))
                    #    break
                    pass
                else:
                    # logging.debug(
                    #    'EM (start) distrib_nb: %d %i/%i llk: %f -- %s, %d',
                    #    self.mu.shape[0], i + 1, it, llk[-1],
                    #    self.name, len(cep))
                    pass

        return llk

    def EM_uniform(self, cep, distrib_nb, iteration_min=3, iteration_max=10,
                   llk_gain=0.01, do_init=True):

        """Expectation-Maximization estimation of the Mixture parameters.

        :param cep: set of feature frames to consider
        :param distrib_nb: number of distributions
        :param iteration_min: minimum number of iterations to perform
        :param iteration_max: maximum number of iterations to perform
        :param llk_gain: gain in term of likelihood, stop the training when the gain is less than this value
        :param do_init: boolean, if True initialize the GMM from the training data

        :return llk: a list of log-likelihoods obtained after each iteration

        """

        llk = []

        if do_init:
            self._init_uniform(cep, distrib_nb)
        accum = copy.deepcopy(self)

        for i in range(0, iteration_max):
            accum._reset()
            # serialize the accum
            accum._serialize()
            llk_acc = np.zeros(1)
            sh = llk_acc.shape
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                llk_acc = np.ctypeslib.as_array(tmp.get_obj())
                llk_acc = llk_acc.reshape(sh)
            
            # E step
            # llk.append(self._expectation_parallel(accum, cep, numThread) / cep.shape[0])
            # self._expectation(accum,cep)
            llk.append(self._expectation(accum, cep) / cep.shape[0])

            # M step
            self._maximization(accum)
            if i > 0:
                gain = llk[-1] - llk[-2]
                if gain < llk_gain and i >= iteration_min:
                    logging.debug(
                        'EM (break) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        self.mu.shape[0], i + 1, iteration_max, gain, self.name,
                        len(cep))
                    break
                else:
                    logging.debug(
                        'EM (continu) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        self.mu.shape[0], i + 1, iteration_max, gain, self.name,
                        len(cep))
            else:
                logging.debug(
                    'EM (start) distrib_nb: %d %i/%i llk: %f -- %s, %d',
                    self.mu.shape[0], i + 1, iteration_max, llk[-1],
                    self.name, len(cep))
        return llk

    def _init_uniform(self, cep, distrib_nb):

        # Load data to initialize the mixture
        self._init(cep)
        cov_tmp = copy.deepcopy(self.invcov)
        nb = cep.shape[0]
        self.w = np.full(distrib_nb, 1.0 / distrib_nb, "d")
        self.cst = np.zeros(distrib_nb, "d")
        self.det = np.zeros(distrib_nb, "d")

        for i in range(0, distrib_nb):
            start = nb // distrib_nb * i
            end = max(start + 10, nb)
            mean = np.mean(cep[start:end, :], axis=0)
            if i == 0:
                self.mu = mean
            else:
                self.mu = np.vstack((self.mu, mean))
                self.invcov = np.vstack((self.invcov, cov_tmp))
        self.cov_var_ctl = 1.0 / copy.deepcopy(self.invcov)

        self._compute_all()

