Train a Universal Background Model
==================================

Universal Background (Gaussian mixture) Models (UBM) are trained via EM algorithm using
the Mixture class from **SIDEKIT**.

UBM are trained using acoustic features that can be extracted on-line or loaded and post-processed from existing HDF5 feature files.
We ackowloedge that UBM training might not be the most efficient as post processing of the acoustic features is performed on-line
(computation of the derivatices, concatenation of the different types of features, normalization) and that iterating over the data 
will be time consuming. However, given the performance of parallel computing and the fact that a large quantity of 
data is not necessary to train good quality models, we chose to use this approach which greatly reduces the feature storage on disk.

1. Training using EM split
--------------------------

Follow::

   ubm = sidekit.Mixture()

   ubm.EM_split(features_server, 
                feature_list, 
                distrib_nb,
                iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8), 
                num_thread=10,
                llk_gain=0.01, 
                save_partial=False,
                ceil_cov=10, 
                floor_cov=1e-2
                )

2. Training using simple EM with fixed number of distributions
--------------------------------------------------------------

follow::
   
   ubm = sidekit.Mixture()

   ubm.EM_uniform(cep, 
                  distrib_nb, 
                  iteration_min=3, 
                  iteration_max=10,
                  llk_gain=0.01, 
                  do_init=True
                  )

3 Full covariance UBM
---------------------

follow::
   
   ubm = sidekit.Mixture()

   ubm.EM_split(features_server, 
                feature_list, 
                distrib_nb,
                iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8), 
                num_thread=10,
                llk_gain=0.01, 
                save_partial=False,
                ceil_cov=10, 
                floor_cov=1e-2
                )

   ubm.EM_convert_full(features_server, 
                       featureList, 
                       distrib_nb,
                       iterations=2, 
                       num_thread=10
                       )



