Run a GMM-UBM system
====================

| The script ``rsr2015_ubm-gmm.py`` run an experiment on the male evaluation part of the RSR2015
| database. The protocols used here is based on the one described in [Larcher14].
| In this version, we only consider the non-target trials where impostors
| pronounce the correct text (Imp Correct).

| The number of Target trials performed is then
| TAR correct: 10,244
| IMP correct: 573,664

[Larcher14] Anthony Larcher, Kong Aik Lee, Bin Ma and Haizhou Li,
"Text-dependent speaker verification: Classifiers, databases and RSR2015,"
in Speech Communication 60 (2014) 56-77


.. topic:: Input/Output
   
   Enter:

     - the number of distribution for the Gaussian Mixture Models
     - the root directory where the RSR2015 database is stored

   Generates the following outputs:
      
      - a :mod:`Mixture` in compressed pickle format (ubm)
      - a :mod:`StatServer` of zero and first-order statistics (enroll_stat)
      - a :mod:`StatServer` containing the super vectors of MAP adapted GMM models for each speaker (enroll_sv)
      - a score file
      - a DET plot


First, loads the required PYTHON packages::

   import numpy as np
   import sidekit
   import os
   import sys
   import multiprocessing
   import matplotlib.pyplot as mpl
   import logging

Enter here your parameters::

   distribNb = 128  # number of Gaussian distributions for each GMM
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

   enroll_idmap = sidekit.IdMap('task/3sesspwd_eval_m_trn.p')
   test_ndx = sidekit.Ndx('task/3sess-pwd_eval_m_ndx.p')
   key = sidekit.Key('task/3sess-pwd_eval_m_key.p')
   with open('task/ubm_list.txt') as inputFile:
       ubmList = inputFile.read().split('\n')
   

Create a `FeaturesServer` to process audio files::
   
   fs = sidekit.FeaturesServer(input_dir=audioDir,
                    input_file_extension='.sph',
                    label_dir='./',
                    label_file_extension='.lbl',
                    from_file='audio',
                    config='sid_16k')


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
speaker GMM models. An empty :mod:`StatServer` is initialized from the ``enroll_idmap`` :mod:`IdMap`. Statistics are then computed
in the ``enroll_stat``  :mod:`StatServer` which is then stored in compressed pickle format::

   # Create a StatServer for the enrollment data and compute the statistics
   enroll_stat = sidekit.StatServer(enroll_idmap)
   enroll_stat.accumulate_stat_parallel(ubm, fs, numThread=nbThread)
   enroll_stat.save_pickle('data/stat_rsr2015_male_enroll.p')
   
Train a GMM for each speaker. Only adapt the mean supervector and store all of them in the ``enrol_sv`` :mod:`StatServer`
that is then stored to disk::

   regulation_factor = 3  # MAP regulation factor
   enroll_sv = enroll_stat.adapt_mean_MAP(ubm, regulation_factor)
   enroll_sv.save_pickle('data/sv_rsr2015_male_enroll.p')


Compute the scores for all trials:: 

   scores_gmm_ubm = sidekit.gmm_scoring(ubm,
                                   enroll_sv,
                                   test_ndx,
                                   fs,
                                   numThread=nbThread)
   scores_gmm_ubm.save_pickle('scores/scores_gmm-ubm_rsr2015_male.p')

   
Plot the Detection Error Trade-off (DET) curve::

   # Set the prior following NIST-SRE 2008 settings
   prior = sidekit.effective_prior(0.01, 10, 1)
   # Initialize the DET plot to 2008 settings
   dp = sidekit.DetPlot(windowStyle='old', plotTitle='GMM-UBM RSR2015 male')
   dp.set_system_from_scores(scores_gmm_ubm, key, sys_name='GMM-UBM')
   dp.create_figure()
   dp.plot_rocch_det(0)
   dp.plot_DR30_both(idx=0)
   dp.plot_mindcf_point(prior, idx=0)

The following plot should be obtained at the end of this tutorial:

.. figure:: rsr2015_GMM-UBM128_map3_snr40_cmvn_rasta_logE.png
