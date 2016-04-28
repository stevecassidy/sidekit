The **libsvm** package
======================

.. topic:: LIBSVM
    
    | is an integrated software 
    | for support vector classification, (C-SVC, nu-SVC), regression (epsilon-SVR, nu-SVR) 
    | and distribution estimation (one-class SVM)
    |
    | **SIDEKIT** only makes use of the library and Python wrapper provided in **LIBSVM**
    | if a fully functional version of the **LIBSVM** library is available in the ``sidekit/libsvm/`` directory
    | The :mod:`libsvm` package released with **SIDEKIT** provides high level interfaces
    | to use Support Vector Machines for speaker recognition.
    | For more details about **LIBSVM** you can refer to the `original website <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_


.. warning::
    **SIDEKIT** requires a version of the libsvm library that is compatible with your machine.
    Before running **SIDEKIT**, download, and compile the libsvm library to make sure you have
    the corresponding file (libsvm.dll for windows and libsvm.so.2 for UNIX-like OS)
    in the ``sidekit/libsvm/`` directory.

.. toctree::
   :maxdepth: 2

   libsvm/libsvm_core.rst
