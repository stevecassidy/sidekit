Quick installation
==================

Dependencies
------------

| **SIDEKIT** requires the installation of the following tools.

   * | Python
     | **SIDEKIT** has been developed under Python 2.7.8 and tested under Python 3.4
 
      - LINUX: python is natively available on most of LINUX distributions
      - OSX: natively available, you can install a different version of python via Homebrew
      - Windows: Python can be installed on Windows through PythonXY, WinPython or anaconda packages

   * pip: to install other required Python packages 

| After installing Python and pip, follow the :ref:`Quick-guide`.

Python packages
---------------

| The following packages are required to use **SIDEKIT**.
| You can install them on your own or follow the procedure 
| described in the :ref:`Quick-guide`.

    - matplotlib==1.3.1
    - mock==1.0.1
    - nose==1.3.4
    - numpy==1.9.0
    - pyparsing==2.0.2
    - python-dateutil==2.2
    - scipy==0.14.0
    - six==1.8.0
    - wsgiref==0.1.2
    - h5py==2.3.1 (optional)


.. _Quick-guide:

Quick guide using virtualenv
----------------------------

| First, be sure to have `virtualenv` installed. 
| You can find some documentation on `the official website <http://virtualenv.readthedocs.org/en/latest/virtualenv.html#installation>`_.
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
|
| **Install the requirements**
| 
| Use the `requirements.txt` provided with the project to install the good requirements:
|
|    ``pip install -r requirements.txt``
|
| You of course need to have your virtualenv activated first.
|
| **Day to day usage**
| 
| Be sure to activate your environment. When you want to add a new dependency to your project, install it using pip like below:
|
|    ``pip install requests``
|
| And then, _freeze_ your new requirements:
|
|    ``pip freeze > requirements.txt``
|
| If you want to deactivate your environment, you just have to ask for it:
|
|    ``deactivate``
|
| Here you go.



Quick guide without virtualenv
------------------------------

| Install ``sidekit`` by using ``pip``
|
|    ``pip install sidekit``
|
| Use the `requirements.txt` provided with the project to install the good requirements:
|
|    ``pip install -r requirements.txt``


Optional linkage
----------------

Those packages might be used by **SIDEKIT** if installed.
To do so, just make sure they are installed on your machine.
When importing, **SIDEKIT** will look for them and link if possible.

   * HDF5

      - LINUX: hdf5 package is available on most of the distributions through package managers (apt, yasp...)
      - | OSX: we recommend to install HDF5 through ``HOMEBREW`` package manager.
        | Since HDF5 has been moved to Homebrew-science, don't forget to tap this directory::
        | 
        | ``brew tap homebrew/science``
        | ``brew install hdf5``

      - Windows: download the HDF5 library and follow the instructions in ``INSTALL_windows``

   * | LibSVM: library dedicated to SVM classifiers. This library can be downloaded from
     | the `official website <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ and easily compiled on all plat-forms
     | Compile the library (``libsvm.so.2`` on UNIX/Linux and Mac platforms and ``libsvm.dll`` on windows)
     | and create a link or copy this library in ``./sidekit/libsvm/``.
