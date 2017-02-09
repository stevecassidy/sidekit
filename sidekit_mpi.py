import copy
import numpy
import multiprocessing
import os
import logging
import h5py
import scipy
import warnings
import sys
from sidekit.statserver import StatServer
from sidekit.factor_analyser import FactorAnalyser
from sidekit.mixture import Mixture
from sidekit.sidekit_io import write_matrix_hdf5, read_matrix_hdf5

from sidekit.sv_utils import serialize
from sidekit.factor_analyser import E_gather, E_worker
from mpi4py import MPI


data_type = numpy.float32

def total_variability(stat_server_file_name,
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
    Train a total variability model using multiple process on multiple nodes with MPI.

    Example of how to train a total variability matrix using MPI.
    Here is what your script should look like:

    ----------------------------------------------------------------

    import sidekit

    fa = sidekit.FactorAnalyser()
    fa.total_variability_mpi("/lium/spk1/larcher/expe/MPI_TV/data/statserver.h5",
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
    comm = MPI.COMM_WORLD

    comm.Barrier()

    # this lines allows to process a single StatServer or a list of StatServers
    if not isinstance(stat_server_file_name, list):
        stat_server_file_name = [stat_server_file_name]

    # Initialize useful variables
    sv_size = ubm.get_mean_super_vector().shape[0]
    gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"
    nb_distrib, feature_size = ubm.mu.shape
    upper_triangle_indices = numpy.triu_indices(tv_rank)

    # Initialize the FactorAnalyser, mean and Sigma are initialized at ZEROS as statistics are centered
    factor_analyser = FactorAnalyser()
    factor_analyser.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
    factor_analyser.F = serialize(numpy.zeros((sv_size, tv_rank)).astype(data_type))
    if tv_init is None:
        factor_analyser.F = numpy.random.randn(sv_size, tv_rank).astype(data_type)
    else:
        factor_analyser.F = tv_init
    factor_analyser.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

    # Save init if required
    if comm.rank == 0:
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        if save_init:
            factor_analyser.write(output_file_name + "_init.h5")

    # Iterative training of the FactorAnalyser
    for it in range(nb_iter):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _A = serialize(numpy.zeros((nb_distrib, tv_rank * (tv_rank + 1) // 2), dtype=data_type))
            _C = serialize(numpy.zeros((tv_rank, sv_size), dtype=data_type))
            _R = serialize(numpy.zeros((tv_rank * (tv_rank + 1) // 2), dtype=data_type))

        if comm.rank == 0:
            total_session_nb = 0

        # E-step
        for stat_server_file in stat_server_file_name:

            with h5py.File(stat_server_file, 'r') as fh:
                nb_sessions = fh["segset"].shape[0]

                if comm.rank == 0:
                    total_session_nb += nb_sessions

                comm.Barrier()
                if comm.rank == 0:
                    logging.critical("Process file: {}".format(stat_server_file))

                # Allocate a list of sessions to process to each node
                local_session_nb = nb_sessions // comm.size
                local_session_idx = numpy.arange(comm.rank * local_session_nb, (comm.rank + 1) * local_session_nb)

                # For each node, divide the sessions to process to create batches
                batch_nb = int(numpy.floor(nb_sessions / float(batch_size) + 0.999))
                batch_indices = numpy.array_split(local_session_idx, batch_nb)

                manager = multiprocessing.Manager()
                q = manager.Queue()
                pool = multiprocessing.Pool(num_thread + 2)

                # put listener to work first
                watcher = pool.apply_async(E_gather, ((_A, _C, _R), q))
                # fire off workers
                jobs = []

                # Load data per batch to reduce the memory footprint
                for batch_idx in batch_indices:
                # Create list of argument for a process
                    arg = fh["stat0"][batch_idx, :], fh["stat1"][batch_idx, :], ubm, factor_analyser.F
                    job = pool.apply_async(E_worker, (arg, q))
                    jobs.append(job)

                # collect results from the workers through the pool result queue
                for job in jobs:
                    job.get()

                # now we are done, kill the listener
                q.put((None, None, None, None))
                pool.close()

                _A, _C, _R = watcher.get()

            comm.Barrier()

        comm.Barrier()

        # Sum all statistics
        if comm.rank == 0:
            # only processor 0 will actually get the data
            total_A = numpy.zeros_like(_A)
            total_C = numpy.zeros_like(_C)
            total_R = numpy.zeros_like(_R)
        else:
            total_A = [None] * _A.shape[0]
            total_C = None
            total_R = None

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

        comm.Barrier()

        # M-step
        if comm.rank == 0:

            total_R /= total_session_nb

            _A_tmp = numpy.zeros((tv_rank, tv_rank), dtype=data_type)
            for c in range(nb_distrib):
                distrib_idx = range(c * feature_size, (c + 1) * feature_size)
                _A_tmp[upper_triangle_indices] = _A_tmp.T[upper_triangle_indices] = total_A[c, :]
                print("total_A[:3,:3] = {}".format(total_A[:3,:3]))
                print("total_C[:3,:3] = {}".format(total_C[:3,:3]))
                factor_analyser.F[distrib_idx, :] = scipy.linalg.solve(_A_tmp, total_C[:, distrib_idx]).T

            # minimum divergence
            if min_div:
                _R_tmp = numpy.zeros((tv_rank, tv_rank), dtype=data_type)
                _R_tmp[upper_triangle_indices] = _R_tmp.T[upper_triangle_indices] = total_R
                ch = scipy.linalg.cholesky(_R_tmp)
                factor_analyser.F = factor_analyser.F.dot(ch)

            # Save the current FactorAnalyser
            if output_file_name is not None:
                if it < nb_iter - 1:
                    factor_analyser.write(output_file_name + "_it-{}.h5".format(it))
                else:
                    factor_analyser.write(output_file_name + ".h5")
                factor_analyser.F = comm.bcast(factor_analyser.F, root=0)

        comm.Barrier()


def extract_ivector(self,
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


def EM_split(ubm,
             comm,
             features_server,
             feature_list,
             distrib_nb,
             output_filename,
             iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8),
             llk_gain=0.01,
             save_partial=False,
             ceil_cov=10,
             floor_cov=1e-2,
             num_thread=1):
    """Expectation-Maximization estimation of the Mixture parameters.

    :param comm:
    :param features: a 2D-array of feature frames (one raow = 1 frame)
    :param distrib_nb: final number of distributions
    :param iterations: list of iteration number for each step of the learning process
    :param llk_gain: limit of the training gain. Stop the training when gain between
            two iterations is less than this value
    :param save_partial: name of the file to save intermediate mixtures,
           if True, save before each split of the distributions
    :param ceil_cov:
    :param floor_cov:

    :return llk: a list of log-likelihoods obtained after each iteration
    """

    if comm.rank == 0:
        # Load the features
        #features = features_server.stack_features_parallel(feature_list, 10)
        features = read_matrix_hdf5(feature_list)

        import sys
        print("size of features: {}".format(sys.getsizeof(features)))

        llk = []

        # Initialize the mixture
        n_frames, feature_size = features.shape
        mu = features.mean(axis=0)
        cov = numpy.mean(features**2, axis=0)
        ubm.mu = mu[None]
        ubm.invcov = 1./cov[None]
        ubm.w = numpy.asarray([1.0])
        ubm.cst = numpy.zeros(ubm.w.shape)
        ubm.det = numpy.zeros(ubm.w.shape)
        ubm.cov_var_ctl = 1.0 / copy.deepcopy(ubm.invcov)
        ubm._compute_all()

    else:
        n_frames = None
        feature_size = None
        features = None

    comm.Barrier()

    # Broadcast the UBM on each process
    ubm = comm.bcast(ubm, root=0)

    # Send n_frames and feature_size to all process
    n_frames = comm.bcast(n_frames, root=0)
    feature_size = comm.bcast(feature_size, root=0)

    # Compute the size of all matrices to scatter to each process
    indices = numpy.array_split(numpy.arange(n_frames), comm.size, axis=0)
    sendcounts = numpy.array([idx.shape[0] * feature_size for idx in indices])
    displacements = numpy.hstack((0, numpy.cumsum(sendcounts)[:-1]))

    # Scatter features on all process
    local_features = numpy.empty((indices[comm.rank].shape[0], feature_size))

    comm.Scatterv([features, tuple(sendcounts), tuple(displacements), MPI.DOUBLE], local_features)
    comm.Barrier()

    # for N iterations:
    for nbg, it in enumerate(iterations[:int(numpy.log2(distrib_nb))]):

        if comm.rank == 0:
            logging.critical("Start training model with {} distributions".format(2**nbg))
            # Save current model before spliting
            if save_partial:
                ubm.write(output_filename + '_{}g.h5'.format(ubm.get_distrib_nb()), prefix='')

        ubm._split_ditribution()
            
        if comm.rank == 0:
            accum = copy.deepcopy(ubm)
        else:
            accum = Mixture()
            accum.w = accum.mu = accum.invcov = None

        # Create one accumulator for each process
        local_accum = copy.deepcopy(ubm)
        for i in range(it):

            local_accum._reset()

            if comm.rank == 0:
                logging.critical("\titeration {} / {}".format(i+1, it))
                _tmp_llk = numpy.array(0)
                accum._reset()
                tmp_w = numpy.zeros_like(ubm.w)
                tmp_mu = numpy.zeros_like(ubm.mu)
                tmp_invcov = numpy.zeros_like(ubm.invcov)

            else:
                _tmp_llk = numpy.array([None])
                tmp_w = None
                tmp_mu = None
                tmp_invcov = None

            # E step
            logging.critical("\nStart E-step, rank {}".format(comm.rank))
            local_llk = numpy.array(ubm._expectation(local_accum, local_features))

            # Reduce all accumulators in process 1
            comm.Barrier()
            comm.Reduce(
                [local_accum.w, MPI.DOUBLE],
                [accum.w, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            comm.Reduce(
                [local_accum.mu, MPI.DOUBLE],
                [accum.mu, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            comm.Reduce(
                [local_accum.invcov, MPI.DOUBLE],
                [accum.invcov, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

            comm.Reduce(
                [local_llk, MPI.DOUBLE],
                [_tmp_llk, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )
            comm.Barrier()
                
            if comm.rank == 0:
                llk.append(_tmp_llk / numpy.sum(accum.w))

                # M step
                logging.critical("\nStart M-step, rank {}".format(comm.rank))
                ubm._maximization(accum, ceil_cov=ceil_cov, floor_cov=floor_cov)

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
             # Send the new Mixture to all process
            comm.Barrier()
            #self.w = comm.bcast(self.w, root=0)
            #self.mu = comm.bcast(self.mu, root=0)
            #self.invcov = comm.bcast(self.invcov, root=0)
            #self.invchol = comm.bcast(self.invchol, root=0)
            #self.cov_var_ctl = comm.bcast(self.cov_var_ctl, root=0)
            #self.cst = comm.bcast(self.cst, root=0)
            #self.det = comm.bcast(self.det, root=0)
            #self.A = comm.bcast(self.A, root=0)
            ubm = comm.bcast(ubm, root=0)
            comm.Barrier()
    ubm.write(output_filename + '_{}g.h5'.format(ubm.get_distrib_nb()), prefix='')
    #return llk

