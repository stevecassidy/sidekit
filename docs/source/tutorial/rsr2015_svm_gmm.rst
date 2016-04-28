
Run a SVM GMM system on the RSR2015 database
============================================

This script run an experiment on the male evaluation part of the
**RSR2015** database. The protocol used here is based on the one
described in [Larcher14]. In this version, we only consider the
non-target trials where impostors pronounce the correct text (Imp
Correct).

The number of Target trials performed is then - TAR correct: 10,244 -
IMP correct: 573,664

[Larcher14] Anthony Larcher, Kong Aik Lee, Bin Ma and Haizhou Li,
"Text-dependent speaker verification: Classifiers, databases and
RSR2015," in Speech Communication 60 (2014) 56–77

Input/Output
------------

Enter:
~~~~~~

-  the number of distribution for the Gaussian Mixture Models
-  the root directory where the RSR2015 database is stored

Generates the following outputs:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  a Mixture in compressed pickle format (ubm)
-  a StatServer of zero and first-order statistics (enroll\_stat)
-  a StatServer of zero and first-order statistics (back\_stat)
-  a StatServer of zero and first-order statistics (nap\_stat)
-  a StatServer of zero and first-order statistics (test\_stat)
-  a StatServer containing the super vectors of MAP adapted GMM models
   for each speaker (enroll\_sv)
-  a StatServer containing the super vectors of MAP adapted GMM models
   for each speaker (back\_sv)
-  a StatServer containing the super vectors of MAP adapted GMM models
   for each speaker (nap\_sv)
-  a StatServer containing the super vectors of MAP adapted GMM models
   for each speaker (test\_sv)
-  a score file
-  a DET plot

Loads the required PYTHON packages:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import numpy as np
    import sidekit
    import multiprocessing
    import os
    import sys
    import matplotlib.pyplot as mpl
    import logging
    
    logging.basicConfig(filename='log/rsr2015_svm-gmm.log',level=logging.INFO)

Set your own parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    distribNb = 4  # number of Gaussian distributions for each GMM
    NAP = False  # activate the Nuisance Attribute Projection
    nap_rank = 40
    rsr2015Path = '/Users/larcher/LIUM/data/RSR2015/RSR2015_V1/'
    
    
    # Default for RSR2015
    audioDir = os.path.join(rsr2015Path , 'sph/male')


Automatically set the number of parallel process to run. The number of
threads to run is set equal to the number of cores available on the
machine minus one or to 1 if the machine has a single core.

.. code:: python

    nbThread = max(multiprocessing.cpu_count()-1, 1)

Load IdMap, Ndx, Key from HDF5 files and ubm\_list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

that define the task. Note that these files are generated when running
``rsr2015_init.py``:

.. code:: python

    print('Load task definition')
    enroll_idmap = sidekit.IdMap('task/3sesspwd_eval_m_trn.h5')
    nap_idmap = sidekit.IdMap('task/3sess-pwd_eval_m_nap.h5')
    back_idmap = sidekit.IdMap('task/3sess-pwd_eval_m_back.h5')
    test_ndx = sidekit.Ndx('task/3sess-pwd_eval_m_ndx.h5')
    test_idmap = sidekit.IdMap('task/3sess-pwd_eval_m_test.h5')
    key = sidekit.Key('task/3sess-pwd_eval_m_key.h5')
    
    with open('task/ubm_list.txt') as inputFile:
        ubmList = inputFile.read().split('\n')

Process the audio to generate MFCC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    print('Initialize FeaturesServers')
    fs = sidekit.FeaturesServer(input_dir=audioDir,
                     input_file_extension='.sph',
                     label_dir='./',
                     label_file_extension='.lbl',
                     from_file='audio',
                     config='sid_16k')    

Train the Universal background Model (UBM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Audio files are process on the fly and acoustic features are not saved
to disk (see SRE10 tutorial to see how to save features to disk). An
empty Mixture is then initialized and an EM algorithm is run to estimate
the UBM before saving it to disk:

.. code:: python

    print('Train the UBM by EM')
    # load all features in a list of arrays
    ubm = sidekit.Mixture()
    llk = ubm.EM_split(fs, ubmList, distribNb, numThread=nbThread)
    ubm.save_pickle('gmm/ubm.p')

Compute the sufficient statistics on the UBM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make use of the new UBM to compute the sufficient statistics of all
enrolement sessions that should be used to train the speaker GMM models,
models for the SVM training blacklist, segments to train the NAP matrix
and test segments. An empty StatServer is initialized. Statistics are
then computed in the StatServer which is then stored to disk:

.. code:: python

    print('Compute the sufficient statistics')
    # Create a StatServer for the enrollment data and compute the statistics
    enroll_stat = sidekit.StatServer(enroll_idmap, ubm)
    enroll_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(enroll_stat.segset.shape[0]), numThread=nbThread)
    enroll_stat.save('data/stat_rsr2015_male_enroll.h5')
    
    back_stat = sidekit.StatServer(back_idmap, ubm)
    back_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(back_stat.segset.shape[0]), numThread=nbThread)
    back_stat.save('data/stat_rsr2015_male_back.h5')
       
    nap_stat = sidekit.StatServer(nap_idmap, ubm)
    nap_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(nap_stat.segset.shape[0]), numThread=nbThread) 
    nap_stat.save('data/stat_rsr2015_male_nap.h5')
       
    test_stat = sidekit.StatServer(test_idmap, ubm)
    test_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(test_stat.segset.shape[0]), numThread=nbThread) 
    test_stat.save('data/stat_rsr2015_male_test.h5')

Train a GMM for each session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only adapt the mean supervector and store all of them in the enrol\_sv
StatServer that is then stored in compressed picked format:

.. code:: python

    print('MAP adaptation of the speaker models')
    regulation_factor = 3  # MAP regulation factor
    
    enroll_sv = enroll_stat.adapt_mean_MAP(ubm, regulation_factor, norm=True)
    enroll_sv.save('data/sv_rsr2015_male_enroll.h5')
    
    back_sv = back_stat.adapt_mean_MAP(ubm, regulation_factor, norm=True)
    back_sv.save('data/sv_rsr2015_male_back.h5')
    
    nap_sv = nap_stat.adapt_mean_MAP(ubm, regulation_factor, norm=True)
    nap_sv.save('data/sv_rsr2015_male_nap.h5')
    
    test_sv = test_stat.adapt_mean_MAP(ubm, regulation_factor, norm=True)
    test_sv.save('data/sv_rsr2015_male_test.h5')

Apply Nuisance Attribute Projection if required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``NAP == True``, estimate and apply the Nuisance Attribute Projection
on all supervectors:

.. code:: python

    if NAP:
        print('Estimate and apply NAP')
        napMat = back_sv.get_nap_matrix_stat1(nap_rank);
        back_sv.stat1 = back_sv.stat1 \
                        - np.dot(np.dot(back_sv.stat1, napMat), napMat.transpose())
        enroll_sv.stat1 = enroll_sv.stat1 \
                        - np.dot(np.dot(enroll_sv.stat1, napMat), napMat.transpose())
        test_sv.stat1 = test_sv.stat1 \
                        - np.dot(np.dot(test_sv.stat1, napMat), napMat.transpose())

Train the Support Vector Machine models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a Support Vector Machine for each speaker by considering the three
sessions of this speaker:

.. code:: python

    print('Train the SVMs')
    sidekit.svm_training('svm/', back_sv, enroll_sv, numThread=nbThread)

Compute all trials and save scores in HDF5 format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the scores for all trials:

.. code:: python

    print('Compute trial scores')
    scores_gmm_svm = sidekit.svm_scoring('svm/', test_sv, test_ndx, numThread=nbThread)
    if NAP:
        scores_gmm_svm.save('scores/scores_svm-gmm_NAP_rsr2015_male.h5')
    else:
        scores_gmm_svm.save('scores/scores_svm-gmm_rsr2015_male.h5')

Plot DET curve and compute minDCF and EER
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    print('Plot the DET curve')
    # Set the prior following NIST-SRE 2008 settings
    prior = sidekit.effective_prior(0.01, 10, 1)
    # Initialize the DET plot to 2008 settings
    dp = sidekit.DetPlot(windowStyle='old', plotTitle='SVM-GMM RSR2015 male')
    dp.set_system_from_scores(scores_gmm_svm, key, sys_name='SVM-GMM')
    dp.create_figure()
    dp.plot_rocch_det(0)
    dp.plot_DR30_both(idx=0)
    dp.plot_mindcf_point(prior, idx=0)

After running this script you should obtain the following curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from IPython.display import Image
    Image(filename='SVM-GMM_128g.png')




.. image:: SVM-GMM_128g.png


