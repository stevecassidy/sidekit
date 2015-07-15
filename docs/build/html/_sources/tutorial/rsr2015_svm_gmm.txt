Run a SVM-GMM system
====================
   
| The script ``rsr2015_svm-gmm.py`` run an experiment on the male evaluation part of the RSR2015
| database. The protocols used here is based on the one described in [Larcher14].
| In this version, we only consider the non-target trials where impostors
| pronounce the correct text (Imp Correct).

| The number of Target trials performed is then
| TAR correct: 10,244
| IMP correct: 573,664

[Larcher14] Anthony Larcher, Kong Aik Lee, Bin Ma and Haizhou Li,
"Text-dependent speaker verification: Classifiers, databases and RSR2015,"
in Speech Communication 60 (2014) 56–77


.. topic:: Input/Output
   
   Enter:

     - the number of distribution for the Gaussian Mixture Models
     - the root directory where the RSR2015 database is stored

   Generates the following outputs:
      
      - a :mod:`Mixture` in compressed pickle format (ubm)
      - a :mod:`StatServer` of zero and first-order statistics (enroll_stat)
      - a :mod:`StatServer` of zero and first-order statistics (back_stat)
      - a :mod:`StatServer` of zero and first-order statistics (nap_stat)
      - a :mod:`StatServer` of zero and first-order statistics (test_stat)
      - a :mod:`StatServer` containing the super vectors of MAP adapted GMM models for each speaker (enroll_sv)
      - a :mod:`StatServer` containing the super vectors of MAP adapted GMM models for each speaker (back_sv)
      - a :mod:`StatServer` containing the super vectors of MAP adapted GMM models for each speaker (nap_sv)
      - a :mod:`StatServer` containing the super vectors of MAP adapted GMM models for each speaker (test_sv)
      - a score file
      - a DET plot

First, loads the required PYTHON packages::

   import numpy as np
   import sidekit
   import multiprocessing
   import os
   import sys
   import matplotlib.pyplot as mpl
   import logging

   logging.basicConfig(filename='rsr2015_ubm-gmm.log',level=logging.INFO)
   
Enter here your parameters::

   distribNb = 128  # number of Gaussian distributions for each GMM
   NAP = True  # activate the Nuisance Attribute Projection
   nap_rank = 40
   rsr2015Path = '/lium/parolee/larcher/data/RSR2015_V1/'
   
Set default parameters for the **RSR2015** database::

   audioDir = os.path.join((rsr2015Path , '/sph/male/')
   
Set the number of process to run in parallel. Default is the number of 
cores of the machine minus one::

   # Automatically set the number of parallel process to run.
   # The number of threads to run is set equal to the number of cores available 
   # on the machine minus one or to 1 if the machine has a single core.
   nbThread = max(multiprocessing.cpu_count()-1, 1)
      
Load :mod:`IdMap`, :mod:`Ndx`, :mod:`Key` objects and lists that define the task.
Note that these files are generated when running ``rsr2015_init.py``::

   print('Load task definition')
   enroll_idmap = sidekit.IdMap('task/3sesspwd_eval_m_trn.p')
   nap_idmap = sidekit.IdMap('task/3sess-pwd_eval_m_nap.p')
   back_idmap = sidekit.IdMap('task/3sess-pwd_eval_m_back.p')
   test_ndx = sidekit.Ndx('task/3sess-pwd_eval_m_ndx.p')
   test_idmap = sidekit.IdMap('task/3sess-pwd_eval_m_test.p')
   key = sidekit.Key('task/3sess-pwd_eval_m_key.p')
   with open('task/ubm_list.txt') as inputFile:
       ubmList = inputFile.read().split('\n')


Load data to train a Universal Background Model.
Audio files are process on the fly and acoustic features 
are not saved to disk (see _`SRE10` tutorial to see how to save features 
to disk). An empty :mod:`Mixture` is then initialized and
an EM algorithm is run to estimate the UBM before saving it to disk::

   data = fs.load_and_stack(np.array(ubmList), nbThread)
   ubm = sidekit.Mixture()
   llk = ubm.EM_split(data, distribNb, numThread=nbThread)
   ubm.save_pickle('gmm/ubm.p')
    
Make use of the new UBM to compute the sufficient statistics of all enrolement sessions that should be used to train the
speaker GMM models, models for the SVM training blacklist, segments to train the NAP matrix and test segments. 
An empty :mod:`StatServer` is initialized. Statistics are then computed
in the :mod:`StatServer` which is then stored to disk::

   # Create a StatServer for the enrollment data and compute the statistics
   enroll_stat = sidekit.StatServer(enroll_idmap)
   enroll_stat.accumulate_stat_parallel(ubm, fs, numThread=nbThread)
   enroll_stat.save_pickle('data/stat_rsr2015_male_enroll.p')

   back_stat = sidekit.StatServer(back_idmap)
   back_stat.accumulate_stat_parallel(ubm, fs, numThread=nbThread)
   back_stat.save_pickle('data/stat_rsr2015_male_back.p')

   nap_stat = sidekit.StatServer(nap_idmap)
   nap_stat.accumulate_stat_parallel(ubm, fs, numThread=nbThread)
   nap_stat.save_pickle('data/stat_rsr2015_male_nap.p')

   test_stat = sidekit.StatServer(test_idmap)
   test_stat.accumulate_stat_parallel(ubm, fs, numThread=nbThread)
   test_stat.save_pickle('data/stat_rsr2015_male_test.p')
   
Train a GMM for each session. Only adapt the mean supervector and store all of them in the ``enrol_sv`` :mod:`StatServer`
that is then stored in compressed picked format::

   regulation_factor = 3  # MAP regulation factor

   enroll_sv = enroll_stat.adapt_mean_MAP(ubm, regulation_factor, norm=True)
   enroll_sv.save_pickle('data/sv_rsr2015_male_enroll.p')

   back_sv = back_stat.adapt_mean_MAP(ubm, regulation_factor, norm=True)
   back_sv.save_pickle('data/sv_rsr2015_male_back.p')

   nap_sv = nap_stat.adapt_mean_MAP(ubm, regulation_factor, norm=True)
   nap_sv.save_pickle('data/sv_rsr2015_male_nap.p')

   test_sv = test_stat.adapt_mean_MAP(ubm, regulation_factor, norm=True)
   test_sv.save_pickle('data/sv_rsr2015_male_test.p')
   
 
If ``NAP == True``, estimate and apply the Nuisance Attribute Projection on all supervectors::

   if NAP:
       print('Estimate and apply NAP')
       napMat = back_sv.get_nap_matrix_stat1(nap_rank);
       back_sv.stat1 = back_sv.stat1 - np.dot(np.dot(back_sv.stat1, napMat), napMat.transpose())
       enroll_sv.stat1 = enroll_sv.stat1 - np.dot(np.dot(enroll_sv.stat1, napMat), napMat.transpose())
       test_sv.stat1 = test_sv.stat1 - np.dot(np.dot(test_sv.stat1, napMat), napMat.transpose())
   
Train a Support Vector Machine for each speaker by considering the three sessions of this speaker::   
   
   sidekit.svm_training('svm/', back_sv, enroll_sv, numThread=nbThread)   
   
Compute the scores for all trials:: 

   print('Compute trial scores')
   scores_gmm_svm = sidekit.svm_scoring('svm/', test_sv, test_ndx, numThread=nbThread)
   if NAP:
       scores_gmm_svm.save_pickle('scores/scores_svm-gmm_NAP_rsr2015_male.p')
   else:
       scores_gmm_svm.save_pickle('scores/scores_svm-gmm_rsr2015_male.p')
   
   
Plot the Detection Error Trade-off (DET) curve::

   # Set the prior following NIST-SRE 2008 settings
   prior = sidekit.effective_prior(0.01, 10, 1)
   # Initialize the DET plot to 2008 settings
   dp = sidekit.DetPlot(windowStyle='old', plotTitle='SVM-GMM RSR2015 male')
   dp.set_system_from_scores(scores_gmm_svm, key, sys_name='SVM-GMM')
   dp.create_figure()
   dp.plot_rocch_det(0)
   dp.plot_DR30_both(idx=0)
   dp.plot_mindcf_point(prior, idx=0)
 

The following plot should be obtained at the end of this tutorial without and with NAP

.. figure:: SVM-GMM_128g.png

.. figure:: SVM-GMM_NAP_128g.png
