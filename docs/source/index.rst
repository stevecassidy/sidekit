.. SIDEKIT documentation master file, created by
   sphinx-quickstart on Mon Oct 27 10:12:02 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |logo| image:: logo_lium.png


Welcome to SIDEKIT 1.1.7 documentation!
=======================================

| **SIDEKIT** is an open source package for Speaker and Language recognition.
| The aim of **SIDEKIT** is to provide an educational and efficient toolkit for speaker/language recognition
| including the whole chain of treatment that goes from the audio data to the analysis of the system performance.


:Authors: 
    Anthony Larcher \&
    Kong Aik Lee \&
    Sylvain Meignier

<<<<<<< HEAD
:Version: 1.1.7 of 2016/11/18

.. seealso::

   News for **SIDEKIT** 1.1.7:
=======
:Version: 1.2 of 2017/02/09

.. seealso::

   News for **SIDEKIT** 1.2:
>>>>>>> dev

      - new ``sidekit_mpi`` module that allows parallel computing on several nodes (cluster)
        MPI implementations are provided for GMM EM algorithm, TotalVariability matrix EM estimation
        and i-vector extraction
        see `MPI <https://pythonhosted.org/mpi4py/>`_ for more information about MPI
      - new ``FactorAnalyser`` class that simplifies the interface
         Note that FA estimation and i-vector extraction is still available in ``StatServer`` but deprecated
      - i-vector scoring with scaling factor
      - uncertainty propagation is available in PLDA scoring


What's here?
============

.. toctree::
   :maxdepth: 1
   :name: mastertoc

   overview/index.rst
   install/index.rst
   api/envvar.rst
   api/index.rst
   tutorial/index.rst
   addon/index.rst


Citation
--------

When using **SIDEKIT** for research, please cite:

| Anthony Larcher, Kong Aik Lee and Sylvain Meignier, 
| **An extensible speaker identification SIDEKIT in Python**,
| in International Conference on Audio Speech and Signal Processing (ICASSP), 2016

Documentation
-------------

This documentation is available in PDF format :download:`here <../build/latex/sidekit.pdf>`
<<<<<<< HEAD

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
=======
>>>>>>> dev


Contacts and info
-----------------

.. toctree::
   :maxdepth: 3
   :titlesonly:

   contact.rst
<<<<<<< HEAD
   aboutSIDEKIT.rst

Additional material
===================

.. toctree::
   :maxdepth: 2

   Links.rst
   references.rst
   known_errors.rst
   datasets.rst
=======
>>>>>>> dev


Sponsors
========

|logo|

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

