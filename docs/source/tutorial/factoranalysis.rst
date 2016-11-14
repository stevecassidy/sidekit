The unified Factor Analyser training
====================================

.. Note::  Until the current version of SIDEKIT, Factor Analysis is a method of the StatServer. This will be changed in future versions.

`factor_analysis` is used to train all types of Factor Analysers including
Total Variability, Joint Factor Analysis and Probabilistic Linear Discriminant Analysis.
The interface for the method is as follow::

   mean, F, G, H, sigma = stat_server.factor_analysis(rank_f,
                                                      rank_g=0,
                                                      rank_h=None,
                                                      re_estimate_residual=False,
                                                      it_nb=(10, 10, 10),
                                                      min_div=True,
                                                      ubm=None,
                                                      batch_size=100,
                                                      num_thread=1,
                                                      save_partial=False,
                                                      init_matrices=(F_init, G_init, H_init))

Where the parameters are:

+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| Option                     | Value (default is bold)                              | Description                                                                                     |
+============================+======================================================+=================================================================================================+
| rank_f                     | integer                                              | Rank of the between class factor matrix                                                         |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| rank_g                     | **0**, integer                                       | Rank of the Within  class factor matrix                                                         |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| rank_h                     | **None**, boolean                                    | if True, estimate the residual covariance matrix. Default is False.                             |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| re_estimate_residual       | **False**, boolean                                   | if True, estimate the residual covariance matrix. Default is False.                             |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| it_nb                      | **(10, 10, 10)**, tuple of three integers            | number of iteration to train between, within class covariance matrices and MAP covariance       |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| min_div                    | **True**, boolean                                    | run minimum divergence re-estimation step after each M-step                                     |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ubm                        | **None**, Mixture                                    | Universal Background Model used to get mean and variance of the data                            |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| batch_size                 | **100**, integer                                     | size of the batch to compute in parallel (used to reduce the memory footprint)                  |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| num_thread                 | **1**, integer                                       | number of parallel process to run                                                               |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| save_partial               | **False**, boolean                                   | if True, save the factor loaded matrices after each iteration                                   |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| init_matrices              | **(None, None, None)**                               | tuple of three matrices used to initialize F, G and H respectiuvely                             |
+----------------------------+------------------------------------------------------+-------------------------------------------------------------------------------------------------+

