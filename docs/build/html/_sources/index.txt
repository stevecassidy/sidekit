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

:Version: 1.0.0 of 2016/02/29 

Implementation
--------------

| **SIDEKIT** has been designed and written in `Python <https://www.python.org>`_ and released under LGPL :ref:`license`
| to allow a wider usage of the code that, we hope, could be beneficial to the community.
| The structure of the core package makes use of a limited number of classes in order
| to facilitate the readability and reusability of the code.
| **SIDEKIT** has been tested under Python 2.7 and Python 3.4 for both Linux and MacOS.


Citation
--------

When using **SIDEKIT** for research, please cite:

| Anthony Larcher, Kong Aik Lee and Sylvain Meignier, 
| **An extensible speaker identification SIDEKIT in Python**,
| in International Conference on Audio Speech and Signal Processing (ICASSP), 2016

Documentation
-------------

| This documentation is available in PDF format :download:`here <sidekit.pdf>`

Download 
--------

| You can download here :download:`SIDEKIT 1.0.0 <SIDEKIT-1.0.0.tar.gz>`
| For the latest version, use GIT and  Pypi. See the :ref:`Install`

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




Contents
========

.. toctree::
   :maxdepth: 3
   :titlesonly:

   aboutSIDEKIT.rst
   howto.rst
   sidekit.rst

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

