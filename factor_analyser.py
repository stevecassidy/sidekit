# -* coding: utf-8 -*-
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
Copyright 2014-2016 Sylvain Meignier and Anthony Larcher

    :mod:`factor_analyser` provides methods to train different types of factor analysers

"""
import copy
import numpy
import multiprocessing
import os
import logging
import h5py
import scipy
import warnings
import ctypes
import sys
from sidekit.statserver import StatServer
from sidekit.mixture import Mixture
from sidekit.sidekit_wrappers import process_parallel_lists, deprecated, check_path_existance
from sidekit.sidekit_io import write_matrix_hdf5
from mpi4py import MPI

data_type = numpy.float64
ct = ctypes.c_double
if data_type == numpy.float32:
    ct = ctypes.c_float


@process_parallel_lists
def fa_model_loop(batch_start,
                  mini_batch_indices,
                  r,
                  phi,
                  sigma,
                  stat0,
                  stat1,
                  e_h,
                  e_hh,
                  num_thread=1):
    """
    :param batch_start: index to start at in the list
    :param mini_batch_indices: indices of the elements in the list (should start at zero)
    :param r: rank of the matrix
    :param phi: factor matrix
    :param sigma: covariance matrix
    :param stat0: matrix of zero order statistics
    :param stat1: matrix of first order statistics
    :param e_h: accumulator
    :param e_hh: accumulator
    :param num_thread: number of parallel process to run
    """
    if sigma.ndim == 2:
        A = phi.T.dot(phi)
        inv_lambda_unique = dict()
        for sess in numpy.unique(stat0[:,0]):
            inv_lambda_unique[sess] = scipy.linalg.inv(sess * A + numpy.eye(A.shape[0]))

    tmp = numpy.zeros((phi.shape[1], phi.shape[1]), dtype=data_type)

    for idx in mini_batch_indices:
        if sigma.ndim == 1:
            inv_lambda = scipy.linalg.inv(numpy.eye(r) + (phi.T * stat0[idx + batch_start, :]).dot(phi))
        else:
            inv_lambda = inv_lambda_unique[stat0[idx + batch_start, 0]]

        aux = phi.T.dot(stat1[idx + batch_start, :])
        numpy.dot(aux, inv_lambda, out=e_h[idx])
        e_hh[idx] = inv_lambda + numpy.outer(e_h[idx], e_h[idx], tmp)

@process_parallel_lists
def fa_distribution_loop(distrib_indices, _A, stat0, batch_start, batch_stop, e_hh, num_thread=1):
    """
    :param distrib_indices: indices of the distributions to iterate on
    :param _A: accumulator
    :param stat0: matrix of zero order statistics
    :param batch_start: index of the first session to process
    :param batch_stop: index of the last session to process
    :param e_hh: accumulator
    :param num_thread: number of parallel process to run
    """
    tmp = numpy.zeros((e_hh.shape[1], e_hh.shape[1]), dtype=data_type)
    for c in distrib_indices:
        _A[c] += numpy.einsum('ijk,i->jk', e_hh, stat0[batch_start:batch_stop, c], out=tmp)
        # The line abov is equivalent to the two lines below:
        # tmp = (E_hh.T * stat0[batch_start:batch_stop, c]).T
        # _A[c] += numpy.sum(tmp, axis=0)


class FactorAnalyser:
    """
    A class to train factor analyser such as total variability models, Joint Factor Analysers or Probabilistic
    Linear Discriminant Analysis (PLDA).

    :attr mean: mean vector
    :attr F: between class matrix
    :attr G: within class matrix
    :attr H: MAP covariance matrix (for Joint Factor Analysis only)
    :attr Sigma: residual covariance matrix
    """

    def __init__(self,
                 input_file_name=None,
                 mean=None,
                 F=None,
                 G=None,
                 H=None,
                 Sigma=None):
        """
        Initialize a Factor Analyser object to None or by reading FactorAnalyser from an HDF5 file.
        When loading fomr a file, other parameters can be provided to overwrite each of the component.

        :param input_file_name: name of the HDF5 file to read from, default is nNone
        :param mean: the mean vector
        :param F: between class matrix
        :param G: within class matrix
        :param H: MAP covariance matrix
        :param Sigma: residual covariance matrix
        """
        if input_file_name is not None:
            fa = FactorAnalyser.read(input_file_name)
            self.mean = fa.mean
            self.F = fa.F
            self.G = fa.G
            self.H = fa.H
            self.Sigma = fa.Sigma
        else:
            self.mean = None
            self.F = None
            self.G = None
            self.H = None
            self.Sigma = None

        if mean is not None:
            self.mean = mean
        if F is not None:
            self.F = F
        if G is not None:
            self.G = G
        if H is not None:
            self.H = H
        if Sigma is not None:
            self.Sigma = Sigma

    @check_path_existance
    def write(self, output_file_name):
        """
        Write a FactorAnalyser object into HDF5 file

        :param output_file_name: the name of the file to write to
        """
        with h5py.File(output_file_name, "w") as fh:
            kind = numpy.zeros(5, dtype="int16")  # FA with 5 matrix
            if self.mean is not None:
                kind[0] = 1
                fh.create_dataset("fa/mean", data=self.mean,
                                  compression="gzip",
                                  fletcher32=True)
            if self.F is not None:
                kind[1] = 1
                fh.create_dataset("fa/f", data=self.F,
                                  compression="gzip",
                                  fletcher32=True)
            if self.G is not None:
                kind[2] = 1
                fh.create_dataset("fa/g", data=self.G,
                                  compression="gzip",
                                  fletcher32=True)
            if self.H is not None:
                kind[3] = 1
                fh.create_dataset("fa/h", data=self.H,
                                  compression="gzip",
                                  fletcher32=True)
            if self.Sigma is not None:
                kind[4] = 1
                fh.create_dataset("fa/sigma", data=self.Sigma,
                                  compression="gzip",
                                  fletcher32=True)
            fh.create_dataset("fa/kind", data=kind,
                              compression="gzip",
                              fletcher32=True)

    @staticmethod
    def read(input_filename):
        """
         Read a generic FactorAnalyser model from a HDF5 file

        :param input_filename: the name of the file to read from

        :return: a FactorAnalyser object
        """
        fa = FactorAnalyser()
        with h5py.File(input_filename, "r") as fh:
            kind = fh.get("fa/kind").value
            if kind[0] != 0:
                fa.mean = fh.get("fa/mean").value
            if kind[1] != 0:
                fa.F = fh.get("fa/f").value
            if kind[2] != 0:
                fa.G = fh.get("fa/g").value
            if kind[3] != 0:
                fa.H = fh.get("fa/h").value
            if kind[4] != 0:
                fa.Sigma = fh.get("fa/sigma").value
        return fa

    def total_variability_single(self,
                                 stat_server,
                                 ubm,
                                 tv_rank,
                                 nb_iter=20,
                                 min_div=True,
                                 tv_init=None,
                                 save_init=False,
                                 output_file_name=None):
        """
        Train a total variability model using a single process on a single node.

        :param stat_server: the StatServer containing data to train the model
        :param ubm: a Mixture object
        :param tv_rank: rank of the total variability model
        :param nb_iter: number of EM iteration
        :param min_div: boolean, if True, apply minimum divergence re-estimation
        :param tv_init: initial matrix to start the EM iterations with
        :param save_init: boolean, if True, save the initial matrix
        :param output_file_name: name of the file where to save the matrix
        """
        assert(isinstance(stat_server, StatServer) and stat_server.validate()), \
            "First argument must be a proper StatServer"
        assert(isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"
        assert(isinstance(tv_rank, int) and (0 < tv_rank <= min(stat_server.stat1.shape))), \
            "tv_rank must be a positive integer less than the dimension of the statistics"
        assert(isinstance(nb_iter, int) and (0 < nb_iter)), "nb_iter must be a positive integer"
 
        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full" 

        # Set useful variables
        nb_sessions, sv_size = stat_server.stat1.shape
        feature_size = ubm.mu.shape[1]
        nb_distrib = ubm.w.shape[0]    

        # Whiten the statistics for diagonal or full models
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # Initialize TV from given data or randomly
        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.F = numpy.random.randn(sv_size, tv_rank) if tv_init is None else tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # Save init if required
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        if save_init:
            self.write(output_file_name + "_init.h5")

        # Estimate  TV iteratively
        for it in range(nb_iter):
            # Create accumulators for the list of models to process
            _A = numpy.zeros((nb_distrib, tv_rank, tv_rank), dtype=data_type)
            _C = numpy.zeros((tv_rank, feature_size * nb_distrib), dtype=data_type)
        
            _R = numpy.zeros((tv_rank, tv_rank), dtype=data_type)
            _r = numpy.zeros(tv_rank, dtype=data_type)

            # E-step:
            index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)

            for sess in range(stat_server.segset.shape[0]):

                inv_lambda = scipy.linalg.inv(numpy.eye(tv_rank) + (self.F.T *
                                                                    stat_server.stat0[sess, index_map]).dot(self.F))

                Aux = self.F.T.dot(stat_server.stat1[sess, :])
                e_h = Aux.dot(inv_lambda)
                e_hh = inv_lambda + numpy.outer(e_h, e_h)
                
                # Accumulate for minimum divergence step
                _r += e_h 
                _R += e_hh

                # Accumulate for M-step
                _C += numpy.outer(e_h, stat_server.stat1[sess, :])
                _A += e_hh * stat_server.stat0[sess][:, numpy.newaxis, numpy.newaxis]

            _r /= nb_sessions 
            _R /= nb_sessions

            # M-step ( + MinDiv si _R n'est pas None)
            for g in range(nb_distrib):
                distrib_idx = range(g * feature_size, (g + 1) * feature_size)
                self.F[distrib_idx, :] = scipy.linalg.solve(_A[g], _C[:, distrib_idx]).T

            # MINIMUM DIVERGENCE STEP
            if min_div:
                ch = scipy.linalg.cholesky(_R)
                self.F = self.F.dot(ch)

            #Save the complete FactorAnalyser (in a single HDF5 file ???)
            if it < nb_iter - 1:
                self.write(output_file_name + "_it-{}.h5".format(it))
            else:
                self.write(output_file_name + ".h5")
     
    def total_variability_parallel(self,
                                   stat_server,
                                   ubm,
                                   tv_rank,
                                   nb_iter=20,
                                   min_div=True,
                                   tv_init=None,
                                   batch_size=1000,
                                   save_init=False,
                                   output_file_name=None,
                                   num_thread=1):
        """
        Train a total variability model using multiple process with MultiProcessing module on a single node.
        This version might not work for Numpy versions higher than 1.10.X due to memory issues
        with Numpy 1.11 and multiprocessing.

        :param stat_server: the StatServer containing data to train the model
        :param ubm: a Mixture object
        :param tv_rank: rank of the total variability model
        :param nb_iter: number of EM iteration
        :param min_div: boolean, if True, apply minimum divergence re-estimation
        :param tv_init: initial matrix to start the EM iterations with
        :param batch_size: size of the minibatch used to reduce the memory footprint
        :param save_init: boolean, if True, save the initial matrix
        :param output_file_name: name of the file where to save the matrix
        :param num_thread: number of parallel process to run
        """
        assert(isinstance(stat_server, StatServer) and stat_server.validate()), \
            "First argument must be a proper StatServer"
        assert(isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"
        assert(isinstance(tv_rank, int) and (0 < tv_rank <= min(stat_server.stat1.shape))), \
            "tv_rank must be a positive integer less than the dimension of the statistics"
        assert(isinstance(nb_iter, int) and (0 < nb_iter)), "nb_iter must be a positive integer"

        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        nb_sessions, sv_size = stat_server.stat1.shape

        # Whiten the statistics for diagonal or full models
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.F = numpy.random.randn(sv_size, tv_rank) if tv_init is None else tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # Save init if required
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        if save_init:
            self.write(output_file_name + "_init.h5")

        # Estimate  TV iteratively
        stat_server.modelset = stat_server.segset
        session_per_model = numpy.ones(stat_server.modelset.shape)

        for it in range(nb_iter):
            # E-step
            print("E_step")
            # _A, _C, _R = stat_server._expectation(V, self.mean, self.Sigma, session_per_model, batch_size, num_thread)
            r = self.F.shape[-1]
            d = int(stat_server.stat1.shape[1] / stat_server.stat0.shape[1])
            C = stat_server.stat0.shape[1]

            # Replicate self.stat0
            index_map = numpy.repeat(numpy.arange(C), d)
            _stat0 = stat_server.stat0[:, index_map]

            # Create accumulators for the list of models to process
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                _A = numpy.zeros((C, r, r), dtype=data_type)
                tmp_A = multiprocessing.Array(ct, _A.size)
                _A = numpy.ctypeslib.as_array(tmp_A.get_obj())
                _A = _A.reshape(C, r, r)

            _C = numpy.zeros((r, d * C), dtype=data_type)
            _R = numpy.zeros((r, r), dtype=data_type)
            _r = numpy.zeros(r, dtype=data_type)

            # Process in batches in order to reduce the memory requirement
            batch_nb = int(numpy.floor(stat_server.segset.shape[0]/float(batch_size) + 0.999))

            for batch in range(batch_nb):
                print("Process batch {}".format(batch))
                batch_start = batch * batch_size
                batch_stop = min((batch + 1) * batch_size, stat_server.segset.shape[0])
                batch_len = batch_stop - batch_start

                # Allocate the memory to save time
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    e_h = numpy.zeros((batch_len, r), dtype=data_type)
                    tmp_e_h = multiprocessing.Array(ct, e_h.size)
                    e_h = numpy.ctypeslib.as_array(tmp_e_h.get_obj())
                    e_h = e_h.reshape(batch_len, r)

                    e_hh = numpy.zeros((batch_len, r, r), dtype=data_type)
                    tmp_e_hh = multiprocessing.Array(ct, e_hh.size)
                    e_hh = numpy.ctypeslib.as_array(tmp_e_hh.get_obj())
                    e_hh = e_hh.reshape(batch_len, r, r)

                # loop on segments
                fa_model_loop(batch_start=batch_start, mini_batch_indices=numpy.arange(batch_len),
                              r=r, phi=self.F, sigma=self.Sigma,
                              stat0=_stat0, stat1=stat_server.stat1,
                              e_h=e_h, e_hh=e_hh, num_thread=num_thread)

                sqr_inv_sigma = 1/numpy.sqrt(self.Sigma)

                # Accumulate for minimum divergence step
                _r += numpy.sum(e_h * session_per_model[batch_start:batch_stop, None], axis=0)
                _R += numpy.sum(e_hh, axis=0)

                _C += e_h.T.dot(stat_server.stat1[batch_start:batch_stop, :]) / sqr_inv_sigma

                # Parallelized loop on the model id's
                fa_distribution_loop(distrib_indices=numpy.arange(C),
                                     _A=_A,
                                     stat0=stat_server.stat0,
                                     batch_start=batch_start,
                                     batch_stop=batch_stop,
                                     e_hh=e_hh,
                                     num_thread=num_thread)

            _r /= session_per_model.sum()
            _R /= session_per_model.shape[0]

            # M-step
            print("M_step")
            for c in range(C):
                distrib_idx = range(c * d, (c+1) * d)
                self.F[distrib_idx, :] = scipy.linalg.solve(_A[c], _C[:, distrib_idx]).T

            # minimum divergence
            if min_div:
                print('applyminDiv reestimation')
                ch = scipy.linalg.cholesky(_R)
                self.F = self.F.dot(ch)

    def total_variability_mpi(self,
                              comm,
                              stat_server_file_name,
                              ubm,
                              tv_rank,
                              nb_iter=20,
                              min_div=True,
                              tv_init=None,
                              output_file_name=None):
        """
        Train a total variability model using multiple process on multiple nodes with MPI.

        Example of how to train a total variability matrix using MPI.
        Here is what your script should look like:

        ----------------------------------------------------------------

        from mpi4py import MPI
        import sidekit

        comm = MPI.COMM_WORLD
        comm.Barrier()

        fa = sidekit.FactorAnalyser()
        fa.total_variability_mpi(comm,
                                 "/lium/spk1/larcher/expe/MPI_TV/data/statserver.h5",
                                 ubm,
                                 tv_rank,
                                 nb_iter=tv_iteration,
                                 min_div=True,
                                 tv_init=tv_new_init2,
                                 output_file_name="data/TV_mpi")

        ----------------------------------------------------------------

        This script should be run using mpirun command (see MPI4PY website for
        more information about how to use it
            http://pythonhosted.org/mpi4py/
        )

            mpirun --hostfile hostfile ./my_script.py

        :param comm: MPI.comm object defining the group of nodes to use
        :param stat_server_file_name: name of the StatServer file to load (make sure you provide absolute path and that
        it is accessible from all your nodes).
        :param ubm: a Mixture object
        :param tv_rank: rank of the total variability model
        :param nb_iter: number of EM iteration
        :param min_div: boolean, if True, apply minimum divergence re-estimation
        :param tv_init: initial matrix to start the EM iterations with
        :param output_file_name: name of the file where to save the matrix
        """
        comm.Barrier()

        # Initialize the FactorAnalyser, mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.F = numpy.random.randn(ubm.get_mean_super_vector().shape[0], tv_rank) if tv_init is None else tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # Load variables on all nodes
        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"
        nb_distrib, feature_size = ubm.mu.shape
        with h5py.File(stat_server_file_name, 'r') as fh:
            nb_sessions = fh["segset"].shape[0]

        # Select different indices on each process and load statistics to process for this process
        tmp_nb_sessions = nb_sessions // comm.size
        session_idx = numpy.arange(comm.rank * tmp_nb_sessions, (comm.rank + 1) * tmp_nb_sessions)
        stat_server = StatServer.read_subset(stat_server_file_name, session_idx)

        # Whiten the statistics
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        for it in range(nb_iter):

            # Create accumulators for the list of models to process
            _A = numpy.zeros((nb_distrib, tv_rank, tv_rank))
            _C = numpy.zeros((tv_rank, feature_size * nb_distrib))
            _R = numpy.zeros((tv_rank, tv_rank))
            _r = numpy.zeros(tv_rank)

            # Loop on the sessions
            for sess in range(tmp_nb_sessions):
                # on calcule E_h
                # on calcule E_hh

                # Replicate self.stat0
                index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)
                inv_lambda = scipy.linalg.inv(numpy.eye(tv_rank) + (self.F.T *
                                                                    stat_server.stat0[sess, index_map]).dot(self.F))

                Aux = self.F.T.dot(stat_server.stat1[sess, :])
                e_h = Aux.dot(inv_lambda)
                e_hh = inv_lambda + numpy.outer(e_h, e_h)

                # Accumulate for minimum divergence step
                _r += e_h
                _R += e_hh

                # Accumulate for M-step
                _C += numpy.outer(e_h, stat_server.stat1[sess, :])
                _A += e_hh * stat_server.stat0[sess][:, numpy.newaxis, numpy.newaxis]

            comm.Barrier()

            # Sum all statistics
            if comm.rank == 0:
                # only processor 0 will actually get the data
                total_A = numpy.zeros_like(_A)
                total_C = numpy.zeros_like(_C)
                total_R = numpy.zeros_like(_R)
                total_r = numpy.zeros_like(_r)
            else:
                total_A = [None] * _A.shape[0]
                total_C = None
                total_R = None
                total_r = None

            # Accumulate _A, using a list in order to avoid limitations of MPI (impossible to reduce matrices bigger
            # than 4GB)
            for ii in range(_A.shape[0]):
                _tmp = copy.deepcopy(_A[ii])
                if comm.rank == 0:
                    _total_A = numpy.zeros_like(total_A[ii])
                else:
                    _total_A = None

                comm.Reduce(
                    [_tmp, MPI.DOUBLE],
                    [_total_A, MPI.DOUBLE],
                    op=MPI.SUM,
                    root=0
                )
                if comm.rank == 0:
                    total_A[ii] = copy.deepcopy(_total_A)

            comm.Reduce(
                [_C, MPI.DOUBLE],
                [total_C, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            comm.Reduce(
                [_R, MPI.DOUBLE],
                [total_R, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            comm.Reduce(
                [_r, MPI.DOUBLE],
                [total_r, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            comm.Barrier()

            # M-step
            # Scatter _A and _C matrices to all process to process the M step
            if comm.rank == 0:
                total_C = numpy.asfortranarray(total_C)

            # Compute the size of all matrices to scatter to each process
            indices = numpy.array_split(numpy.arange(nb_distrib), comm.size, axis=0)
            sendcounts = numpy.array([idx.shape[0] for idx in indices])
            displacements = numpy.hstack((0, numpy.cumsum(sendcounts)[:-1]))

            # Create local ndarrays on each process
            _A_local = numpy.zeros((len(indices[comm.rank]), tv_rank, tv_rank))
            _C_local = numpy.zeros((tv_rank, feature_size * len(indices[comm.rank])), order='F')

            # Scatter the accumulators for M step
            comm.Scatterv([total_A, tuple(sendcounts * tv_rank**2), (displacements * tv_rank**2), MPI.DOUBLE], _A_local)
            comm.Scatterv([total_C, tuple(sendcounts * feature_size * tv_rank), tuple(displacements * feature_size * tv_rank), MPI.DOUBLE], _C_local)

            comm.Barrier()

            # F is re-initialized to zero before M-step
            self.F.fill(0.)

            for idx in range(sendcounts[comm.rank]):
                g = displacements[comm.rank] + idx
                distrib_idx = range(g * feature_size, (g + 1) * feature_size)
                local_distrib_idx = range(idx * feature_size, (idx + 1) * feature_size)
                self.F[distrib_idx, :] = scipy.linalg.solve(_A_local[idx], _C_local[:, local_distrib_idx]).T

            comm.Barrier()
            if comm.rank == 0:
                _F = numpy.zeros_like(self.F)
            else:
                _F = None

            # Reduce to get the final self.F in process 0
            comm.Reduce(
                [self.F, MPI.DOUBLE],
                [_F, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            if comm.rank == 0:
                self.F = _F
                total_r /= nb_sessions
                total_R /= nb_sessions

                # min div
                if min_div:
                    ch = scipy.linalg.cholesky(total_R)
                    self.F = self.F.dot(ch)

                # Save the current FactorAnalyser
                if output_file_name is not None:
                    if it < nb_iter - 1:
                        self.write(output_file_name + "_it-{}.h5".format(it))
                    else:
                        self.write(output_file_name + ".h5")
            self.F = comm.bcast(self.F, root=0)

            comm.Barrier()

    def extract_ivectors_single(self,
                                stat_server,
                                ubm,
                                uncertainty=False):
        """
        Estimate i-vectors for a given StatServer using single process on a single node.

        :param stat_server: sufficient statistics
        :param ubm: Mixture object (the UBM)
        :param uncertainty: boolean, if True, return a matrix with uncertainty matrices (diagonal of the matrices)

        :return: a StatServer with i-vectors in the stat1 attribute and a matrix of uncertainty matrices (optional)
        """
        assert(isinstance(stat_server, StatServer) and stat_server.validate()), \
            "First argument must be a proper StatServer"
        assert(isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"

        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        tv_rank = self.F.shape[1]
        feature_size = ubm.mu.shape[1]
        nb_distrib = ubm.w.shape[0]

        # Whiten the statistics for diagonal or full models
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        # Extract i-vectors
        iv_stat_server = StatServer()
        iv_stat_server.modelset = copy.deepcopy(stat_server.modelset)
        iv_stat_server.segset = copy.deepcopy(stat_server.segset)
        iv_stat_server.start = copy.deepcopy(stat_server.start)
        iv_stat_server.stop = copy.deepcopy(stat_server.stop)
        iv_stat_server.stat0 = numpy.ones((stat_server.modelset.shape[0], 1))
        iv_stat_server.stat1 = numpy.ones((stat_server.modelset.shape[0], tv_rank))

        iv_sigma = numpy.ones((stat_server.modelset.shape[0], tv_rank))

        # Replicate self.stat0
        index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)

        for sess in range(stat_server.segset.shape[0]):

            inv_lambda = scipy.linalg.inv(numpy.eye(tv_rank) + (self.F.T *
                                                                stat_server.stat0[sess, index_map]).dot(self.F))
            Aux = self.F.T.dot(stat_server.stat1[sess, :])
            iv_stat_server.stat1[sess, :] = Aux.dot(inv_lambda)
            iv_sigma[sess, :] = inv_lambda + numpy.outer(iv_stat_server.stat1[sess, :], iv_stat_server.stat1[sess, :])

        if uncertainty:
            return iv_stat_server, iv_sigma
        else:
            return iv_stat_server

    def extract_ivector_mp(self):
        """
        Parallel extraction of i-vectors using multiprocessing module
        This version might not work for Numpy versions higher than 1.10.X due to memory issues
        with Numpy 1.11 and multiprocessing.
        """
        pass

    def extract_ivector_mpi(self,
                            comm,
                            stat_server_file_name,
                            ubm,
                            output_file_name,
                            uncertainty=False,
                            prefix=''):
        """
        Estimate i-vectors for a given StatServer using multiple process on multiple nodes.

        :param comm: MPI.comm object defining the group of nodes to use
        :param stat_server_file_name: file name of the sufficient statistics StatServer HDF5 file
        :param ubm: Mixture object (the UBM)
        :param output_file_name: name of the file to save the i-vectors StatServer in HDF5 format
        :param uncertainty: boolean, if True, saves a matrix with uncertainty matrices (diagonal of the matrices)
        :param prefix: prefixe of the dataset to read from in HDF5 file
        """
        assert(isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"

        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        tv_rank = self.F.shape[1]
        feature_size = ubm.mu.shape[1]
        nb_distrib = ubm.w.shape[0]

        # Get the number of sessions to process
        with h5py.File(stat_server_file_name, 'r') as fh:
            nb_sessions = fh["segset"].shape[0]

        # Work on each node with different data
        indices = numpy.array_split(numpy.arange(nb_sessions), comm.size, axis=0)
        sendcounts = numpy.array([idx.shape[0] * self.F.shape[1]  for idx in indices])
        displacements = numpy.hstack((0, numpy.cumsum(sendcounts)[:-1]))

        stat_server = StatServer.read_subset(stat_server_file_name, indices[comm.rank])

        # Whiten the statistics for diagonal or full models
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        # Estimate i-vectors
        if comm.rank == 0:
            iv = numpy.zeros((nb_sessions, tv_rank))
            iv_sigma = numpy.zeros((nb_sessions, tv_rank))
        else:
            iv = None
            iv_sigma = None

        local_iv = numpy.zeros((stat_server.modelset.shape[0], tv_rank))
        local_iv_sigma = numpy.ones((stat_server.modelset.shape[0], tv_rank))

        # Replicate self.stat0
        index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)

        for sess in range(stat_server.segset.shape[0]):

             inv_lambda = scipy.linalg.inv(numpy.eye(tv_rank) + (self.F.T * stat_server.stat0[sess, index_map]).dot(self.F))

             Aux = self.F.T.dot(stat_server.stat1[sess, :])
             local_iv[sess, :] = Aux.dot(inv_lambda)
             local_iv_sigma[sess, :] = numpy.diag(inv_lambda + numpy.outer(local_iv[sess, :], local_iv[sess, :]))
        comm.Barrier()

        comm.Gatherv(local_iv,[iv, sendcounts, displacements,MPI.DOUBLE], root=0)
        comm.Gatherv(local_iv_sigma,[iv_sigma, sendcounts, displacements,MPI.DOUBLE], root=0)

        if comm.rank == 0:

            with h5py.File(stat_server_file_name, 'r') as fh:
                iv_stat_server = StatServer()
                iv_stat_server.modelset = fh.get(prefix+"modelset").value
                iv_stat_server.segset = fh.get(prefix+"segset").value

                # if running python 3, need a conversion to unicode
                if sys.version_info[0] == 3:
                    iv_stat_server.modelset = iv_stat_server.modelset.astype('U', copy=False)
                    iv_stat_server.segset = iv_stat_server.segset.astype('U', copy=False)

                tmpstart = fh.get(prefix+"start").value
                tmpstop = fh.get(prefix+"stop").value
                iv_stat_server.start = numpy.empty(fh[prefix+"start"].shape, '|O')
                iv_stat_server.stop = numpy.empty(fh[prefix+"stop"].shape, '|O')
                iv_stat_server.start[tmpstart != -1] = tmpstart[tmpstart != -1]
                iv_stat_server.stop[tmpstop != -1] = tmpstop[tmpstop != -1]
                iv_stat_server.stat0 = numpy.ones((nb_sessions, 1))
                iv_stat_server.stat1 = iv

            iv_stat_server.write(output_file_name)
            if uncertainty:
                path = os.path.splitext(output_file_name)
                write_matrix_hdf5(iv_sigma, path[0] + "_uncertainty" + path[1])

    def plda(self,
             stat_server,
             rank_f,
             nb_iter=10,
             scaling_factor=1.,
             output_file_name=None,
             save_partial=False):
        """
        Train a simplified Probabilistic Linear Discriminant Analysis model (no within class covariance matrix
        but full residual covariance matrix)

        :param stat_server: StatServer object with training statistics
        :param rank_f: rank of the between class covariance matrix
        :param nb_iter: number of iterations to run
        :param scaling_factor: scaling factor to downscale statistics (value bewteen 0 and 1)
        :param output_file_name: name of the output file where to store PLDA model
        :param save_partial: boolean, if True, save PLDA model after each iteration
        """
        vect_size = stat_server.stat1.shape[1]

        # Initialize mean and residual covariance from the training data
        self.mean = stat_server.get_mean_stat1()
        self.Sigma = stat_server.get_total_covariance_stat1()

        # Sum stat per model
        model_shifted_stat, session_per_model = stat_server.sum_stat_per_model()
        class_nb = model_shifted_stat.modelset.shape[0]

        # Multiply statistics by scaling_factor
        model_shifted_stat.stat0 *= scaling_factor
        model_shifted_stat.stat1 *= scaling_factor
        session_per_model *= scaling_factor

        # Compute Eigen Decomposition of Sigma in order to initialize the EigenVoice matrix
        sigma_obs = stat_server.get_total_covariance_stat1()
        evals, evecs = scipy.linalg.eigh(sigma_obs)
        idx = numpy.argsort(evals)[::-1]
        evecs = evecs.real[:, idx[:rank_f]]
        self.F = evecs[:, :rank_f]

        # Estimate PLDA model by iterating the EM algorithm
        for it in range(nb_iter):
            logging.info('Estimate between class covariance, it %d / %d', it + 1, nb_iter)

            # E-step
            print("E_step")

            # Copy stats as they will be whitened with a different Sigma for each iteration
            local_stat = copy.deepcopy(model_shifted_stat)

            # Whiten statistics (with the new mean and Sigma)
            local_stat.whiten_stat1(self.mean, self.Sigma)

            # Whiten the EigenVoice matrix
            eigen_values, eigen_vectors = scipy.linalg.eigh(self.Sigma)
            ind = eigen_values.real.argsort()[::-1]
            eigen_values = eigen_values.real[ind]
            eigen_vectors = eigen_vectors.real[:, ind]
            sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values.real)
            sqr_inv_sigma = numpy.dot(eigen_vectors, numpy.diag(sqr_inv_eval_sigma))
            self.F = sqr_inv_sigma.T.dot(self.F)

            # Replicate self.stat0
            index_map = numpy.zeros(vect_size, dtype=int)
            _stat0 = local_stat.stat0[:, index_map]

            e_h = numpy.zeros((class_nb, rank_f))
            e_hh = numpy.zeros((class_nb, rank_f, rank_f))

            # loop on model id's
            fa_model_loop(batch_start=0,
                          mini_batch_indices=numpy.arange(class_nb),
                          r=rank_f,
                          phi=self.F,
                          sigma=self.Sigma,
                          stat0=_stat0,
                          stat1=local_stat.stat1,
                          e_h=e_h,
                          e_hh=e_hh,
                          num_thread=1)

            # Accumulate for minimum divergence step
            _R = numpy.sum(e_hh, axis=0) / session_per_model.shape[0]

            _C = e_h.T.dot(local_stat.stat1).dot(scipy.linalg.inv(sqr_inv_sigma))
            _A = numpy.einsum('ijk,i->jk', e_hh, local_stat.stat0.squeeze())

            # M-step
            self.F = scipy.linalg.solve(_A, _C).T

            # Update the residual covariance
            self.Sigma = sigma_obs - self.F.dot(_C) / session_per_model.sum()


            # Minimum Divergence step
            self.F = self.F.dot(scipy.linalg.cholesky(_R))

            if output_file_name is None:
                output_file_name = "temporary_plda"

            if save_partial and it < nb_iter - 1:
                self.write(output_file_name + "_it-{}.h5".format(it))
            elif it == nb_iter - 1:
                self.write(output_file_name + ".h5")
