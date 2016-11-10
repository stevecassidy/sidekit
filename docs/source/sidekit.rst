API description
===============

.. automodule:: sidekit

    :Authors:
        Anthony LACHER, Sylvain MEIGNIER \& Kong Aik LEE

    :Version: 1.1.6 of 2016/10/31

| This package is the core of the **SIDEKIT** toolkit.
| While developing **SIDEKIT**, we tried to keep in mind two
| targets:
|
|     1. limit the number of classes to allow better readability
|     2. make **SIDEKIT** compatible with existing tools
|
| To reach this target, we have created four Main Classes
| (:mod:`FeaturesExtractor``, :mod:`FeaturesServer`, :mod:`Mixture` and :mod:`StatServer`) which can be used 
| together with a number of tools available in companion
| modules (:mod:`sidekit_io` and :mod:`sv_utils`).
|
| Front-end and back-end processing such as acoustic feature extraction 
| and score analysis are handled in two packages: :mod:`frontend` and :mod:`bosaris`.
| The :mod:`frontend` package include a number of tools to extract and normalize
| the acoustic features as well as detecting the high energy frames for voice 
| activity detection. The :mod:`bosaris` package consists of the translation of 
| a part of the BOSARIS toolkit available on `this webpage <https://sites.google.com/site/bosaristoolkit/>`_.
| The current python implementation of the BOSARIS toolkit does not include tools
| for calibration and fusion but only the core structures that are used to manage
| enrollment lists, trial definitions and scores.
| The authors would like to thank Niko Brummer and AGNITIO to allow them to distribute 
| this version of the BOSARIS toolkit.

.. note::
    The bosaris package which is released together with **SIDEKIT** is distributed under a different
    license. The intellectual property belongs to the original authors of the toolkit.
 


.. toctree::
   :maxdepth: 2
   :titlesonly:

   sidekit_classes.rst
   sidekit_modules.rst
   bosaris.rst
   frontend.rst
   libsvm.rst

