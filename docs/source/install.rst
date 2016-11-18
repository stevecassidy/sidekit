.. _Install:

Using Conda (recommended)
=========================

After installing `miniconda <http://conda.pydata.org/miniconda.html>`_:

.. code-block:: bash

   conda install -c anthol sidekit


Using PIP
=========

.. code-block:: bash

   pip install sidekit

In a Virtual environment
========================


| First, be sure to have `virtualenv` installed.
| You can find some documentation on `the official website <http://virtualenv.readthedocs.org/en/latest/>`_.
|
| **Create your virtual environment**
|
|    ``virtualenv env``
|
| This will create a directory called ``env`` in the current directory.
| If you want to specify a different python interpreter (for example to test you program with python 3),
| you just have to use the `-p` option:
|
|    ``virtualenv -p /path/to/python3 env``
|
| **Activate your environment**
|
| Each and every time you will want to work on your project, you will have to first activate your virtualenv:
|
|    ``. ./env/bin/activate``
|
| Your prompt should change and you should see the name of your virtualenv between ``()``. In our case ``(env)``.


Dependencies
============

| **SIDEKIT** requires the installation of the following tools.

   * | Python
     | **SIDEKIT** has been developed under Python 3.3, 3.4 and 3.5

      - LINUX: python is natively available on most of LINUX distributions
      - OSX: natively available, you can install a different version of python via Homebrew
      - Windows: Python can be installed on Windows through PythonXY, WinPython or anaconda packages

   * To install other required Python packages use one of the following:
      - conda
      - pip

| The following packages are required to use **SIDEKIT**.
| You can install them on your own or follow the procedure
| described in the :ref:`Quick-guide`.

    - matplotlib
    - mock==1.0.1
    - nose==1.3.4
    - numpy
    - pyparsing==2.0.2
    - python-dateutil==2.2
    - scipy
    - six==1.8.0
    - wsgiref==0.1.2
    - h5py==2.3.1
    - pandas
    - theano

Optional linkage
----------------

Those packages might be used by **SIDEKIT** if installed.
To do so, just make sure they are installed on your machine.
When importing, **SIDEKIT** will look for them and link if possible.


   * | LibSVM: library dedicated to SVM classifiers. This library can be downloaded from
     | the `official website <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ and easily compiled on all plat-forms
     | Compile the library (``libsvm.so.2`` on UNIX/Linux and Mac platforms and ``libsvm.dll`` on windows)
     | and create a link or copy this library in ``./sidekit/libsvm/``.
