.. SIDEKIT documentation master file, created by
   sphinx-quickstart on Mon Oct 27 10:12:02 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |logo| image:: logo_lium.png


Welcome to SIDEKIT’s documentation!
===================================

| **SIDEKIT** is an open source package for Speaker and Language recognition.
| The aim of **SIDEKIT** is to provide an educational and efficient toolkit 
| for speaker/language recognition including the whole chain of treatment 
| that goes from the audio data to the analysis of the system performance.


:Authors: 
    Anthony Larcher \&
    Kong Aik Lee \&
    Sylvain Meignier

:Version: 1.1.6 of 2016/10/31

.. seealso::

   News for **SIDEKIT** 1.1.6:

      - new ``FeaturesExtractor`` for a simplified interface
      - new ``FeaturesServer`` provides a simpler and more flexible management of the acoustic parameters
      - HDF5 is now used to store acoustic features to reduce storage and allow more flexibility

.. warning::

   Parallel computation using ``multiprocessing`` does not support the latest **Numpy 1.11**
   In order to benefit from ``multiprocessing`` faster computation, keep using **Numpy <1.11**.
   Next version of **SIDEKIT** will come with `MPI <https://pythonhosted.org/mpi4py/>`_ implementation
   to allow parallel computation on a single node but also on **MULTIPLE NODE** (see `dev` branch of the GIT
   repository for a `beta` version.

Implementation
--------------

| **SIDEKIT** has been designed and written in `Python <https://www.python.org>`_ and released under LGPL :ref:`license`
| to allow a wider usage of the code that, we hope, could be beneficial to the community.
| The structure of the core package makes use of a limited number of classes in order
| to facilitate the readability and reusability of the code.
| Starting from version 1.1.0 SIDEKIT is no longer tested under Python 2.* In case you want to keep using Python2, you may have modification to do on your own.
| **SIDEKIT** has been tested under Python >3.3 for both Linux and MacOS.


Citation
--------

When using **SIDEKIT** for research, please cite:

| Anthony Larcher, Kong Aik Lee and Sylvain Meignier, 
| **An extensible speaker identification SIDEKIT in Python**,
| in International Conference on Audio Speech and Signal Processing (ICASSP), 2016

Documentation
-------------

This documentation is available in PDF format :download:`here <sidekit.pdf>`

Download and Install
--------------------

All you need to get **SIDEKIT** on your machine.
It is possible to get the sources to manually include in your PYTHONPATH or you can install
via **pip** or **conda**.

.. toctree::
   :maxdepth: 1
   :name: mastertoc

   download.rst
   install.rst



What for
--------

| **SIDEKIT** aims at providing the whole chain of tools required to perform speaker recognition.
| The main tools available include:

   * Acoustic features extraction

      - Linear-Frequency Cepstral Coefficients (LFCC)
      - Mel-Frequency Cepstral Coefficients (MFCC)
      - RASTA filtering
      - Energy-based Voice Activity Detection (VAD)
      - normalization (CMS, CMVN, Short Term Gaussianization)

   * Modeling and classification
   
      - Gaussian Mixture Models (GMM)
      - *i* - vectors
      - Probabilistic Linear Discriminant Analysis (PLDA)
      - Joint Factor Analysis (JFA)
      - Support Vector Machine (SVM)
      - Deep Neural Network (bridge to THEANO)

   * Presentation of the results
      - DET plot
      - ROC Convex Hull based DET plot


Tutorials
=========

See now how to start with **SIDEKIT** with some basic tutorials and advanced evaluations on standard databases.

.. toctree::
   :maxdepth: 2

   tutorial/shorttuto.rst
   tutorial/tutorial.rst

.. include:: sidekit.rst

Contacts and info
=================

.. toctree::
   :maxdepth: 3
   :titlesonly:

   contact.rst
   aboutSIDEKIT.rst

Additional material
===================

.. toctree::
   :maxdepth: 2

   Links.rst
   references.rst
   known_errors.rst


Sponsors
========

|logo|

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

