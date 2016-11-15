import copy
import numpy
import multiprocessing
import logging
import os
import h5py
import sidekit 
import scipy
import warnings
import ctypes
import random 
from sidekit.sidekit_wrappers import process_parallel_lists, deprecated, check_path_existance
from mpi4py import MPI

data_type = numpy.float64
ct = ctypes.c_double
if data_type == numpy.float32:
    ct = ctypes.c_float


@process_parallel_lists
def fa_model_loop(batch_start,
                  mini_batch_indices,
                  r,
                  phi_white,
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
    :param phi_white: whitened version of the factor matrix
    :param phi: non-whitened version of the factor matrix
    :param sigma: covariance matrix
    :param stat0: matrix of zero order statistics
    :param stat1: matrix of first order statistics
    :param e_h: accumulator
    :param e_hh: accumulator
    :param num_thread: number of parallel process to run
    """
    tmp = numpy.zeros((phi_white.shape[1], phi_white.shape[1]), dtype=data_type)

    for idx in mini_batch_indices:
        if sigma.ndim == 1:
            inv_lambda = scipy.linalg.inv(numpy.eye(r) + (phi_white.T * stat0[idx + batch_start, :]).dot(phi_white))
        else:
            inv_lambda = scipy.linalg.inv(stat0[idx + batch_start, 0] * A + numpy.eye(A.shape[0]))

        Aux = phi_white.T.dot(stat1[idx + batch_start, :])
        numpy.dot(Aux, inv_lambda, out=e_h[idx])
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

    """

    def __init__(self,
                 input_file_name=None,
                 mean=None,
                 F=None,
                 G=None,
                 H=None,
                 Sigma=None):
        """

        """
        self.mean = None
        self.F = None
        self.G = None
        self.H = None
        self.Sigma = None

        if input_file_name is not None:
            fa = FactorAnalyser.read(input_file_name)
            self.mean = fa.mean
            self.F = fa.F
            self.G = fa.G
            self.H = fa.H
            self.Sigma = fa.Sigma

    def TotalVariability_single(self,
                                stat_server,
                                ubm,
                                tv_rank,
                                nb_iter=20,
                                min_div=True,
                                tv_init=None, 
                                save_init=False,
                                output_file_name=None):
        """
        Parameter verification
            - first param is a statserver
            - second param is a Mixture, diag or full ?
            - third param is a positiv integer, less than the size of the input data
            - fourth param is an positive integer
        """
        assert(isinstance(stat_server, sidekit.StatServer) and stat_server.validate()), \
            "First argument must be a proper StatServer"
        assert(isinstance(ubm, sidekit.Mixture) and ubm.validate()), "Second argument must be a proper Mixture"
        assert(isinstance(tv_rank, int) and (0 < tv_rank <= min(stat_server.stat1.shape))), \
            "tv_rank must be a positive integer less than the dimension of the statistics"
        assert(isinstance(nb_iter, int) and (0 < nb_iter)), "nb_iter must be a positive integer"
 
        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full" 

        # Set useful variables
        nb_sessions, sv_size = stat_server.stat1.shape
        feature_size = ubm.mu.shape[1]
        nb_distrib = ubm.w.shape[0]    
 
        """
        Whiten the statistics
            - for diagonal or full models
        """
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        """ 
        Initialize TV
            - from given data
            - randomly
        
        mean and Sigma are initialized at ZEROS as statistics are centered
        """
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.F = numpy.random.randn(sv_size, tv_rank) if tv_init is None else tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # Save init if required
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        self.write(output_file_name + "_init.h5")

        """
        Estimate  TV iteratively
            dans cette version on bouclera sur les shows et pas les modeles donc pas besoin de sommer par modele
            ou de modifier le statserver

            Creer les accumulateurs E_h, E_hh, _A et _C, _R, _r
        """
        for it in range(nb_iter):
            # Create accumulators for the list of models to process
            _A = numpy.zeros((nb_distrib, tv_rank, tv_rank), dtype=data_type)
            _C = numpy.zeros((tv_rank, feature_size * nb_distrib), dtype=data_type)
        
            _R = numpy.zeros((tv_rank, tv_rank), dtype=data_type)
            _r = numpy.zeros(tv_rank, dtype=data_type)

            """
            E-step:
            """
            # Replicate self.stat0
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

            """
                M-step ( + MinDiv si _R n'est pas None)
                    V = model_shifted_stat._maximization(V, _A, _C, _R)[0]
            """
            for g in range(nb_distrib):
                distrib_idx = range(g * feature_size, (g + 1) * feature_size)
                self.F[distrib_idx, :] = scipy.linalg.solve(_A[g], _C[:, distrib_idx]).T

            # MINIMUM DIVERGENCE STEP
            if min_div:
                ch = scipy.linalg.cholesky(_R)
                self.F = self.F.dot(ch)

            """
                Save the complete FactorAnalyser (in a single HDF5 file ???)
            """
            if it < nb_iter - 1:
                self.write(output_file_name + "_it-{}.h5".format(it))
            else:
                self.write(output_file_name + ".h5")
     
    def total_variability_mp(self,
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
        ancienne version nettoyee, il faut la tester avec numpy 1.11 et voir ce qui ne marche plus
        :return:
        """
        assert(isinstance(stat_server, sidekit.StatServer) and stat_server.validate()), \
            "First argument must be a proper StatServer"
        assert(isinstance(ubm, sidekit.Mixture) and ubm.validate()), "Second argument must be a proper Mixture"
        assert(isinstance(tv_rank, int) and (0 < tv_rank <= min(stat_server.stat1.shape))), \
            "tv_rank must be a positive integer less than the dimension of the statistics"
        assert(isinstance(nb_iter, int) and (0 < nb_iter)), "nb_iter must be a positive integer"

        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        nb_sessions, sv_size = stat_server.stat1.shape
        feature_size = ubm.mu.shape[1]
        nb_distrib = ubm.w.shape[0]

        """
        Whiten the statistics
            - for diagonal or full models
        """
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        """
        Initialize TV
            - from given data
            - randomly

        mean and Sigma are initialized at ZEROS as statistics are centered
        """
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.F = numpy.random.randn(sv_size, tv_rank) if tv_init is None else tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        # Save init if required
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        if save_init:
            self.write(output_file_name + "_init.h5")

        """
        Estimate  TV iteratively
            dans cette version on bouclera sur les shows et pas les modeles donc pas besoin de sommer par modele
            ou de modifier le statserver

            Creer les accumulateurs E_h, E_hh, _A et _C, _R, _r
        """
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

                # loop on model id's
                fa_model_loop(batch_start=batch_start, mini_batch_indices=numpy.arange(batch_len),
                              r=r, phi_white=self.F, sigma=self.Sigma,
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

            if not min_div:
                _R = None

            # M-step
            print("M_step")
            # V = stat_server._maximization(V, _A, _C, _R)[0]
            for c in range(C):
                distrib_idx = range(c * d, (c+1) * d)
                self.F[distrib_idx, :] = scipy.linalg.solve(_A[c], _C[:, distrib_idx]).T

            # MINIMUM DIVERGENCE STEP
            if _R is not None:
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
                              save_init=False,
                              output_file_name=None):
        """
        version fonctionnant sur plusieurs noeuds
        :return:
        """

        comm.Barrier()

        """ 
        Initialize TV
            - from given data
            - randomly
        
        mean and Sigma are initialized at ZEROS as statistics are centered
        """
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape)
        self.F = numpy.random.randn(sv_size, tv_rank) if tv_init is None else tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape)

        """
        Ici on recupere toutes les variables partagees par tous les noeuds:
            - TV
            - nb_sessions
            - nombre de distributions
            - dimension des trames
            - accumulateur _A, _C, _R et _r qui sont sommes a la fin de l etape E
        """
        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"
        nb_distrib, feature_size = ubm.mu.shape
        with h5py.File(stat_server_file_name, 'r') as fh:
            nb_sessions = fh["segset"].shape[0]

        """
        Ici on travaille sur chaque noeud avec des donnees differentes
        """
        # On charge les statistiques pour ce noeud
        tmp_nb_sessions = nb_sessions // comm.size
        session_idx = numpy.arange(comm.rank * tmp_nb_sessions, (comm.rank + 1) * tmp_nb_sessions)
        stat_server = sidekit.StatServer.read_subset(stat_server_file_name, session_idx)

        # On blanchit les stats
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

            # Pour chaque session:
            for sess in range(tmp_nb_sessions):
                # on calcule E_h
                # on calcule E_hh
                # Replicate self.stat0
                logging.critical("iteration {}, node: {}, session {} /{}".format(it, comm.rank, sess, tmp_nb_sessions))
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
 
            """
            Ici on accumule le résultat de chaque noeud dans le noeud 0
            """
            # the 'totals' array will hold the sum of each 'data' array
            if comm.rank == 0:
                # only processor 0 will actually get the data
                total_A = numpy.zeros_like(_A)
                total_C = numpy.zeros_like(_C)
                total_R = numpy.zeros_like(_R)
                total_r = numpy.zeros_like(_r)
            else:
                total_A = None
                total_C = None
                total_R = None
                total_r = None

            # use MPI to get the totals 
            comm.Reduce(
                [_A, MPI.DOUBLE],
                [total_A, MPI.DOUBLE],
                op=MPI.SUM,
                root=0
            )

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

            """
            Etape M sur le noeud 0
            """
            if comm.rank == 0:

                total_nb_sessions = nb_sessions * comm.size
                total_r /= nb_sessions
                total_R /= nb_sessions

                # M
                for g in range(nb_distrib):
                    distrib_idx = range(g * feature_size, (g + 1) * feature_size)
                    self.F[distrib_idx, :] = scipy.linalg.solve(total_A[g], total_C[:, distrib_idx]).T

                # min div
                if min_div:
                    ch = scipy.linalg.cholesky(total_R)
                    self.F = self.F.dot(ch)

                """
                On sauve le resultat de l iteration
                """
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

        :param stat_server:
        :param ubm:
        :param uncertainty:
        :return:
        """
        assert(isinstance(stat_server, sidekit.StatServer) and stat_server.validate()), \
            "First argument must be a proper StatServer"
        assert(isinstance(ubm, sidekit.Mixture) and ubm.validate()), "Second argument must be a proper Mixture"

        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        tv_rank = self.F.shape[1]
        nb_sessions, sv_size = stat_server.stat1.shape
        feature_size = ubm.mu.shape[1]
        nb_distrib = ubm.w.shape[0]

        """
        Whiten the statistics
            - for diagonal or full models
        """
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        """
        Extract i-vectors
        """
        iv_stat_server = sidekit.StatServer()
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

        """
        pass

    def extract_ivector_mpi(self,
                            comm,
                            stat_server_file_name,
                            ubm,
                            uncertainty=False):
        """

        :param comm:
        :param stat_server_file_name:
        :param ubm:
        :param uncertainty:
        :return:
        """
        assert(isinstance(ubm, sidekit.Mixture) and ubm.validate()), "Second argument must be a proper Mixture"
        logging.critical("top 1 - {}".format(comm.rank))
        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        tv_rank = self.F.shape[1]
        feature_size = ubm.mu.shape[1]
        nb_distrib = ubm.w.shape[0]

        # Get the number of sessions to process
        with h5py.File(stat_server_file_name, 'r') as fh:
            nb_sessions = fh["segset"].shape[0]
        logging.critical("top 2 - {}".format(comm.rank))

        """
        Ici on travaille sur chaque noeud avec des donnees differentes
        """
        # On charge les statistiques pour ce noeud
        tmp_nb_sessions = nb_sessions // comm.size
        session_idx = numpy.arange(comm.rank * tmp_nb_sessions, (comm.rank + 1) * tmp_nb_sessions)
        stat_server = sidekit.StatServer.read_subset(stat_server_file_name, session_idx)
        logging.critical("top 3 - {}".format(comm.rank))

        """
        Whiten the statistics
            - for diagonal or full models
        """
        if gmm_covariance == "diag":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), 1. / ubm.get_invcov_super_vector())
        elif gmm_covariance == "full":
            stat_server.whiten_stat1(ubm.get_mean_super_vector(), ubm.invchol)

        """
        Extract i-vectors
        """
        if comm.rank == 0:
            iv = numpy.zeros((stat_server.modelset.shape[0] * comm.size, tv_rank))
            iv_sigma = numpy.zeros((stat_server.modelset.shape[0] * comm.size, tv_rank))
            logging.critical("taile de iv= {}".format(iv.shape))

            # local_iv = None
            # local_iv_sigma = None

            # iv = numpy.zeros((stat_server.modelset.shape[0], tv_rank))
            # iv_sigma = numpy.zeros((stat_server.modelset.shape[0], tv_rank))
        else:
            iv = None
            iv_sigma = None

            local_iv = numpy.zeros((stat_server.modelset.shape[0], tv_rank))
            local_iv_sigma = numpy.ones((stat_server.modelset.shape[0], tv_rank))
            logging.critical("taille de local_iv = {}".format(local_iv.shape))

        if not comm.rank == 0:
            # Replicate self.stat0
            index_map = numpy.repeat(numpy.arange(nb_distrib), feature_size)

            for sess in range(stat_server.segset.shape[0]):

                inv_lambda = scipy.linalg.inv(numpy.eye(tv_rank) + (self.F.T *
                                                                    stat_server.stat0[sess, index_map]).dot(self.F))

                Aux = self.F.T.dot(stat_server.stat1[sess, :])
                local_iv[sess, :] = Aux.dot(inv_lambda)
                local_iv_sigma[sess, :] = numpy.diag(inv_lambda + numpy.outer(local_iv[sess, :], local_iv[sess, :]))
                logging.critical("rank {} - session {} / {}".format(comm.rank, sess, stat_server.segset.shape[0]))
        comm.Barrier()

        local_iv = None
        local_iv_sigma = None

        logging.critical("rank {}, local_iv.shape = {}".format(comm.rank, local_iv.shape))

        comm.Gather(local_iv, iv, root=0)
        logging.critical("rank {}, local_iv.shape = {}".format(comm.rank, local_iv.shape))
        comm.Gather(local_iv_sigma, iv_sigma, root=0)
        logging.critical("rank {}, local_iv.shape = {}".format(comm.rank, local_iv.shape))

        if comm.rank == 0:

            logging.critical("type of iv: {}".format(type(iv)))
            logging.critical("shape of iv: {}".format(iv.shape))

            logging.critical("stat1: type = {}, size = {}, type modelset = {}".format(type(tmp_stat1), len(tmp_stat1),
                                                                                      type(stat_server.modelset)))

            logging.critical("elt 0: {}".format(type(tmp_stat1[0])))

            iv_stat_server = sidekit.StatServer()
            iv_stat_server.modelset = copy.deepcopy(stat_server.modelset)
            iv_stat_server.segset = copy.deepcopy(stat_server.segset)
            iv_stat_server.start = copy.deepcopy(stat_server.start)
            iv_stat_server.stop = copy.deepcopy(stat_server.stop)
            iv_stat_server.stat0 = numpy.ones((stat_server.modelset.shape[0], 1))
            iv_stat_server.stat1 = tmp_stat1[:stat_server.modelset.shape[0]]

            if uncertainty:
                return iv_stat_server, tmp_sigma[:stat_server.modelset.shape[0]]
            else:
                return iv_stat_server

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
         Read a generic FA model from a HDF5 file

        :param input_filename: the name of the file to read from

        :return: a tuple of 5 elements: the mean vector, the between class covariance matrix,
            the within class covariance matrix, the MAP matrix and the residual covariancematrix
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

