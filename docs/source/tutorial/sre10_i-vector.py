# -*- coding: utf-8 -*-
"""
Created on Tue May 12 2015

@author: Anthony Larcher

This script runs an experiment on the male NIST Speaker Recognition
Evaluation 2010 extended core task.
For more details about the protocol, refer to
http://www.itl.nist.gov/iad/mig/tests/sre/2010/

"""


import sys
import numpy as np
import scipy

import os
import copy

import sidekit
import multiprocessing
import matplotlib.pyplot as mpl
import logging
logging.basicConfig(filename='log/sre10_i-vector.log', level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)


expe_root_dir = '/lium/parolee/larcher/src/python/sidekit_tutorial/nist-sre/'
task_dir = '/lium/parolee/larcher/src/python/sidekit_tutorial/nist-sre/task'
i4U_dir = '/lium/parolee/larcher/src/python/sidekit_tutorial/nist-sre/Sph_MetaData'


if not os.path.exists(expe_root_dir):
    os.makedirs(expe_root_dir)
os.chdir(expe_root_dir)

log_dir = os.path.join(expe_root_dir, 'log')
gmm_dir = os.path.join(expe_root_dir, 'gmm')
data_dir = os.path.join(expe_root_dir, 'data')
scores_dir = os.path.join(expe_root_dir, 'scores')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(gmm_dir):
    os.makedirs(gmm_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(scores_dir):
    os.makedirs(scores_dir)

#################################################################
# Set your own parameters
#################################################################
train = True  # if True, train th UBM, TV matrux and extract the i-vectors
test = True  # if True, perform the experiments xonsidering all i-vectors already exist
plot = True  # if True, plot the DET curve with all systems in condition 5
distribNb = 1024  # number of Gaussian distributions for the UBM
rank_TV = 400  # Rank of the Total Variability matrix
audioDir = '/lium/parolee/larcher/data/nist/'  # Root directory where features are stored
# list of scoring to run on the task, could be 'cosine', 'mahalanobis', '2cov' or 'plda'
scoring = ['cosine', 'mahalanobis', '2cov', 'plda']

# Automatically set the number of parallel process to run.
# The number of threads to run is set equal to the number of cores available 
# on the machine minus one or to 1 if the machine has a single core.
nbThread = max(multiprocessing.cpu_count()-1, 1)

#################################################################
# Load IdMap, Ndx, Key from PICKLE files and ubm_list
#################################################################
print('Load task definition')
enroll_idmap = sidekit.IdMap('task/sre10_coreX-coreX_m_trn.h5', 'hdf5')
nap_idmap = sidekit.IdMap('task/sre04050608_m_training.h5', 'hdf5')
back_idmap = sidekit.IdMap('task/sre10_coreX-coreX_m_back.h5', 'hdf5')
test_ndx = sidekit.Ndx('task/sre10_coreX-coreX_m_ndx.h5', 'hdf5')
test_idmap = sidekit.IdMap('task/sre10_coreX-coreX_m_test.h5', 'hdf5')
keys = []
for cond in range(9):
    keys.append(sidekit.Key('task/sre10_coreX-coreX_det{}_key.h5'.format(cond + 1)))

with open('task/ubm_list.txt', 'r') as inputFile:
    ubmList = inputFile.read().split('\n')

if train:
    #################################################################
    # Process the audio to generate MFCC
    #################################################################
    print('Create the feature server to extract MFCC features')
    fs = sidekit.FeaturesServer(input_dir=audioDir,
                                input_file_extension='.mfcc',
                                label_dir='./',
                                label_file_extension='.lbl',
                                from_file='spro4',
                                config='sid_8k',
                                keep_all_features=False)

    #################################################################
    # Train the Universal background Model (UBM)
    #################################################################
    print('Train the UBM by EM')
    data = fs.load_and_stack(np.array(ubmList), nbThread)
    ubm = sidekit.Mixture()
    llk = ubm.EM_split(data, distribNb, numThread=nbThread)
    ubm.save_pickle('gmm/ubm.p')

    #################################################################
    # Compute the sufficient statistics on the UBM
    #################################################################
    print('Compute the sufficient statistics')
    # Create a StatServer for the enrollment data and compute the statistics
    enroll_stat = sidekit.StatServer(enroll_idmap)
    enroll_stat.accumulate_stat_parallel(ubm, fs, numThread=nbThread)
    enroll_stat.save('data/stat_sre10_coreX-coreX_m_enroll.h5')

    nap_stat = sidekit.StatServer(nap_idmap)
    nap_stat.accumulate_stat_parallel(ubm, fs, numThread=nbThread)
    nap_stat.save('data/stat_sre04050608_m_training.h5')

    test_stat = sidekit.StatServer(test_idmap)
    test_stat.accumulate_stat_parallel(ubm, fs, numThread=nbThread)
    test_stat.save('data/stat_sre10_coreX-coreX_m_test.h5')

    enroll_stat = sidekit.StatServer('data/stat_sre10_coreX-coreX_m_enroll.h5')
    nap_stat = sidekit.StatServer('data/stat_sre04050608_m_training.h5')
    test_stat = sidekit.StatServer('data/stat_sre10_coreX-coreX_m_test.h5')

    #################################################################
    # Train Total Variability Matrix for i-vector extration
    #
    # run 10 iterations of the EM algorithm including
    # minimum divergence step
    #################################################################
    print('Estimate Total Variability Matrix')
    mean, TV, G, H, Sigma = nap_stat.factor_analysis(rank_TV,
                                                     itNb=(10, 0, 0), minDiv=True, ubm=ubm,
                                                     batch_size=1000, numThread=nbThread)

    sidekit.sidekit_io.write_pickle(TV, 'data/TV_sre04050608_m.p')
    sidekit.sidekit_io.write_pickle(mean, 'data/TV_mean_sre04050608_m.p')
    sidekit.sidekit_io.write_pickle(Sigma, 'data/TV_Sigma_sre04050608_m.p')

    #################################################################
    # Extract i-vectors for target models, training and test segments
    #################################################################
    print('Extraction of i-vectors') 
    enroll_iv = enroll_stat.estimate_hidden(mean, Sigma, V=TV, U=None, D=None, numThread=nbThread)[0]
    enroll_iv.save('data/iv_sre10_coreX-coreX_m_enroll.h5')

    test_iv = test_stat.estimate_hidden(mean, Sigma, V=TV, U=None, D=None, numThread=nbThread)[0]
    test_iv.save('data/iv_sre10_coreX-coreX_m_test.h5')

    nap_iv = nap_stat.estimate_hidden(mean, Sigma, V=TV, U=None, D=None, numThread=nbThread)[0]
    nap_iv.save('data/iv_sre04050608_m_training.h5')


if test:

    enroll_iv = sidekit.StatServer('data/iv_sre10_coreX-coreX_m_enroll.h5')
    nap_iv = sidekit.StatServer('data/iv_sre04050608_m_training.h5')
    test_iv = sidekit.StatServer('data/iv_sre10_coreX-coreX_m_test.h5')

    #%% 
    #################################################################
    # Compute all trials and save scores in HDF5 format
    #################################################################
    if 'cosine' in scoring:

        print('Run Cosine scoring evaluation without WCCN')
        scores_cos = sidekit.iv_scoring.cosine_scoring(enroll_iv, test_iv, test_ndx, wccn = None)
        scores_cos.save('scores/scores_cosine_sre10_coreX-coreX_m.h5')
    
        print('Run Cosine scoring evaluation with WCCN')
        wccn = nap_iv.get_wccn_choleski_stat1()
        scores_cos_wccn = sidekit.iv_scoring.cosine_scoring(enroll_iv, test_iv, test_ndx, wccn=wccn)
        scores_cos_wccn.save('scores/scores_cosine_wccn_sre10_coreX-coreX_m.h5')
    
        print('Run Cosine scoring evaluation with LDA')
        LDA = nap_iv.get_lda_matrix_stat1(150)
    
        nap_iv_lda = copy.deepcopy(nap_iv)
        enroll_iv_lda = copy.deepcopy(enroll_iv)
        test_iv_lda = copy.deepcopy(test_iv)
    
        nap_iv_lda.rotate_stat1(LDA)
        enroll_iv_lda.rotate_stat1(LDA)
        test_iv_lda.rotate_stat1(LDA)
    
        scores_cos_lda = sidekit.iv_scoring.cosine_scoring(enroll_iv_lda, test_iv_lda, test_ndx, wccn=None)
        scores_cos_lda.save('scores/scores_cosine_lda_sre10_coreX-coreX_m.h5')
    
        print('Run Cosine scoring evaluation with LDA + WCCN')
        wccn = nap_iv_lda.get_wccn_choleski_stat1()
        scores_cos_lda_wcnn = sidekit.iv_scoring.cosine_scoring(enroll_iv_lda, test_iv_lda, test_ndx, wccn=wccn)
        scores_cos_lda_wcnn.save('scores/scores_cosine_lda_wccn_sre10_coreX-coreX_m.h5')


    if 'mahalanobis' in scoring:

        print('Run Mahalanobis scoring evaluation with 1 iteration EFR')
        meanEFR, CovEFR = nap_iv.estimate_spectral_norm_stat1(3)

        nap_iv_efr1 = copy.deepcopy(nap_iv)
        enroll_iv_efr1 = copy.deepcopy(enroll_iv)
        test_iv_efr1 = copy.deepcopy(test_iv)
    
        nap_iv_efr1.spectral_norm_stat1(meanEFR[:1], CovEFR[:1])
        enroll_iv_efr1.spectral_norm_stat1(meanEFR[:1], CovEFR[:1])
        test_iv_efr1.spectral_norm_stat1(meanEFR[:1], CovEFR[:1])
        M1 = nap_iv_efr1.get_mahalanobis_matrix_stat1()
        scores_mah_efr1 = sidekit.iv_scoring.mahalanobis_scoring(enroll_iv_efr1, test_iv_efr1, test_ndx, M1)
        scores_mah_efr1.save('scores/scores_mahalanobis_efr1_sre10_coreX-coreX_m.h5') 


    if '2cov' in scoring:

        print('Run 2Cov scoring evaluation without normalization')
        W = nap_iv.get_within_covariance_stat1()
        B = nap_iv.get_between_covariance_stat1()
        scores_2cov = sidekit.iv_scoring.two_covariance_scoring(enroll_iv, test_iv, test_ndx, W, B)
        scores_2cov.save('scores/scores_2cov_sre10_coreX-coreX_m.h5')
    
        print('Run 2Cov scoring evaluation with 1 iteration of Spherical Norm')
        meanSN, CovSN = nap_iv.estimate_spectral_norm_stat1(1, 'sphNorm')

        nap_iv_sn1 = copy.deepcopy(nap_iv)
        enroll_iv_sn1 = copy.deepcopy(enroll_iv)
        test_iv_sn1 = copy.deepcopy(test_iv)
    
        nap_iv_sn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        enroll_iv_sn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        test_iv_sn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])

        W1 = nap_iv_sn1.get_within_covariance_stat1()
        B1 = nap_iv_sn1.get_between_covariance_stat1()
        scores_2cov_sn1 = sidekit.iv_scoring.two_covariance_scoring(enroll_iv_sn1, test_iv_sn1, test_ndx, W1, B1)
        scores_2cov_sn1.save('scores/scores_2cov_sn1_sre10_coreX-coreX_m.h5')


    if 'plda' in scoring:

        print('Run PLDA scoring evaluation without normalization')    

        meanSN, CovSN = nap_iv.estimate_spectral_norm_stat1(1, 'efr')

        nap_iv_sn1 = copy.deepcopy(nap_iv)
        enroll_iv_sn1 = copy.deepcopy(enroll_iv)
        test_iv_sn1 = copy.deepcopy(test_iv)

        nap_iv_sn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        enroll_iv_sn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        test_iv_sn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])

        nap = copy.deepcopy(nap_iv)
        nap_sn = copy.deepcopy(nap_iv_sn1)

        print('Run PLDA rank = 400, 10 iterations without normalization'.format(rk, it))
        mean, F, G, H, Sigma = nap.factor_analysis(rk, rank_G=0,
                        re_estimate_residual=True,
                        itNb=(it,0,0), minDiv=True, ubm=None,
                        batch_size=1000, numThread=nbThread)
        print('scoring')
        scores_plda = sidekit.iv_scoring.PLDA_scoring(enroll_iv, test_iv, test_ndx,
                                              mean, F, G, Sigma)
        scores_plda.save('scores/scores_plda_rank400_it10_sre10_coreX-coreX_m.h5')

        print('Run PLDA rank = 400, 10 iterations with 1 iteration of Eigen Factor Radial')
        mean1, F1, G1, H1, Sigma1 = nap_sn.factor_analysis(rk, rank_G=0,
                        re_estimate_residual=True,
                        itNb=(it,0,0), minDiv=True, ubm=None,
                        batch_size=1000, numThread=nbThread)
        scores_plda_efr1 = sidekit.iv_scoring.PLDA_scoring(enroll_iv_sn1, test_iv_sn1, test_ndx, mean1, F1, G1, Sigma1)
        scores_plda_efr1.save('scores/scores_plda_rank_400_it10_efr1_sre10_coreX-coreX_m.h5')


if plot:
    print('Plot the DET curve')
    # Set the prior following NIST-SRE 2010 settings
    prior = sidekit.effective_prior(0.001, 1, 1)
    # Initialize the DET plot to 2010 settings
    dp = sidekit.DetPlot(windowStyle='sre10', plotTitle='I-Vectors SRE 2010-ext male, cond 5')

    dp.set_system_from_scores(scores_cos, keysX[4], sys_name='Cosine')
    dp.set_system_from_scores(scores_cos_wccn, keysX[4], sys_name='Cosine WCCN')
    dp.set_system_from_scores(scores_cos_lda, keysX[4], sys_name='Cosine LDA')
    dp.set_system_from_scores(scores_cos_wccn_lda, keysX[4], sys_name='Cosine WCCN LDA')

    dp.set_system_from_scores(scores_mah_efr1, keysX[4], sys_name='Mahalanobis EFR')

    dp.set_system_from_scores(scores_2cov, keysX[4], sys_name='2 Covariance')
    dp.set_system_from_scores(scores_2cov_sn1, keysX[4], sys_name='2 Covariance Spherical Norm')

    dp.set_system_from_scores(scores_plda, keysX[4], sys_name='PLDA')
    dp.set_system_from_scores(scores_plda_efr, keysX[4], sys_name='PLDA EFR')

    dp.create_figure()
    dp.plot_rocch_det(0)
    dp.plot_rocch_det(1)
    dp.plot_rocch_det(2)
    dp.plot_rocch_det(3)
    dp.plot_rocch_det(4)
    dp.plot_rocch_det(5)
    dp.plot_rocch_det(6)
    dp.plot_rocch_det(7)
    dp.plot_rocch_det(8)
    dp.plot_DR30_both(idx=0)
    dp.plot_mindcf_point(prior, idx=0)


