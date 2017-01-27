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
from sidekit.factor_analyser import FactorAnalyser
from sidekit.mixture import Mixture
from sidekit.sidekit_wrappers import process_parallel_lists, deprecated, check_path_existance
from sidekit.sidekit_io import write_matrix_hdf5
from mpi4py import MPI

def total_variability_mpi(stat_server_file_name,
                          ubm,
                          tv_rank,
                          nb_iter=20,
                          min_div=True,
                          tv_init=None,
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

    # Initialize the FactorAnalyser, mean and Sigma are initialized at ZEROS as statistics are centered
    factor_analyser = FactorAnalyser()
    factor_analyser.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
    factor_analyser.F = numpy.random.randn(ubm.get_mean_super_vector().shape[0], tv_rank) if tv_init is None else tv_init
    factor_analyser.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

    # Load variables on all nodes
    gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"
    nb_distrib, feature_size = ubm.mu.shape

    # Iterative training of the FactorAnalyser
    for it in range(nb_iter):
        # Create accumulators for the list of models to process
        _A = serialize(numpy.zeros((nb_distrib, tv_rank * (tv_rank + 1) // 2), dtype=data_type))
        _C = serialize(numpy.zeros((tv_rank, feature_size * nb_distrib), dtype=data_type))
        _R = serialize(numpy.zeros((tv_rank * (tv_rank + 1) // 2), dtype=data_type))

        if comm.rank == 0:
            total_session_nb = 0

        for ssfn in stat_server_file_name:

            with h5py.File(ssfn, 'r') as fh:
                nb_sessions = fh["segset"].shape[0]

                if comm.rank == 0:
                    total_session_nb += nb_sessions

                comm.Barrier()
                if comm.rank == 0:
                    logging.critical("Process file: {}".format(ssfn))

                # Allocate a list of sessions to process to each node
                local_session_nb = nb_sessions // comm.size
                local_session_idx = numpy.arange(comm.rank * local_nb_sessions, (comm.rank + 1) * local_nb_sessions)

                # The job is parallelized on each node by using a Pool of workers
                batch_nb = int(numpy.floor(local_session_nb / float(batch_size) + 0.999))
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
                    arg = fh["stat0"][batch_idx, :], fh["stat1"][batch_idx, :], ubm, self.F
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
        # Scatter _A and _C matrices to all process to process the M step
        #if comm.rank == 0:
        #    total_C = numpy.asfortranarray(total_C)

        # Compute the size of all matrices to scatter to each process
        #indices = numpy.array_split(numpy.arange(nb_distrib), comm.size, axis=0)
        #sendcounts = numpy.array([idx.shape[0] for idx in indices])
        #displacements = numpy.hstack((0, numpy.cumsum(sendcounts)[:-1]))

        # Create local ndarrays on each process
        #_A_local = numpy.zeros((len(indices[comm.rank]), tv_rank, tv_rank))
        #_C_local = numpy.zeros((tv_rank, feature_size * len(indices[comm.rank])), order='F')

        # Scatter the accumulators for M step
        # comm.Scatterv([total_A, tuple(sendcounts * tv_rank**2), (displacements * tv_rank**2), MPI.DOUBLE], _A_local)
        #comm.Barrier()

        #if comm.rank == 0:
        #    for ii, distrib in enumerate(indices[comm.rank]):
        #        _A_local[ii] = copy.deepcopy(total_A[distrib])

        # loop on receiving node
        #for rank in range(1, comm.size):

        #    for ii, distrib in enumerate(indices[rank]):
        #        if comm.rank == 0:
        #            comm.Send([total_A[distrib], MPI.DOUBLE], dest=rank, tag=ii)
        #        elif comm.rank == rank:
        #            comm.Recv([_A_local[ii], MPI.DOUBLE], source=0, tag=ii)

        #comm.Barrier()
        #comm.Scatterv(
        #    [total_C, tuple(sendcounts * feature_size * tv_rank), tuple(displacements * feature_size * tv_rank),
        #     MPI.DOUBLE], _C_local)

        #comm.Barrier()

        # F is re-initialized to zero before M-step
        #factor_analyser.F.fill(0.)

        #for idx in range(sendcounts[comm.rank]):
        #    g = displacements[comm.rank] + idx
        #    distrib_idx = range(g * feature_size, (g + 1) * feature_size)
        #    local_distrib_idx = range(idx * feature_size, (idx + 1) * feature_size)
        #    factor_analyser.F[distrib_idx, :] = scipy.linalg.solve(_A_local[idx], _C_local[:, local_distrib_idx]).T

        # M-step
        _A_tmp = numpy.zeros((tv_rank, tv_rank), dtype=data_type)
        for c in range(nb_distrib):
            distrib_idx = range(c * feature_size, (c + 1) * feature_size)
            _A_tmp[upper_triangle_indices] = _A_tmp.T[upper_triangle_indices] = _A[c, :]
            self.F[distrib_idx, :] = scipy.linalg.solve(_A_tmp, _C[:, distrib_idx]).T

        # minimum divergence
        if min_div:
            _R_tmp = numpy.zeros((tv_rank, tv_rank), dtype=data_type)
            _R_tmp[upper_triangle_indices] = _R_tmp.T[upper_triangle_indices] = _R
            ch = scipy.linalg.cholesky(_R_tmp)
            self.F = self.F.dot(ch)

        # Save the current FactorAnalyser
        if output_file_name is not None:
            if it < nb_iter - 1:
                self.write(output_file_name + "_it-{}.h5".format(it))
            else:

        comm.Barrier()
        if comm.rank == 0:
            logging.critical("after M step")
            _F = numpy.zeros_like(factor_analyser.F)
        else:
            _F = None

        comm.Reduce(
            [factor_analyser.F, MPI.DOUBLE],
            [_F, MPI.DOUBLE],
            op=MPI.SUM,
            root=0
        )

        if comm.rank == 0:
            factor_analyser.F = _F
            total_r /= nb_sessions
            total_R /= nb_sessions

            # min div
            if min_div:
                ch = scipy.linalg.cholesky(total_R)
                factor_analyser.F = factor_analyser.F.dot(ch)

            # Save the current FactorAnalyser
            if output_file_name is not None:
                if it < nb_iter - 1:
                    factor_analyser.write(output_file_name + "_it-{}.h5".format(it))
                else:
                    factor_analyser.write(output_file_name + ".h5")
                factor_analyser.F = comm.bcast(factor_analyser.F, root=0)

        comm.Barrier()