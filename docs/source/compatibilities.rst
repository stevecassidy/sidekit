Compatibilities
===============


| The implementation of **SIDEKIT** benefits from the experience of existing tools and toolkits
| in the community. The main ones are `ALIZE <http://alize.univ-avignon.fr/>`_, `BOSARIS <https://sites.google.com/site/bosaristoolkit/>`_, `HTK <http://htk.eng.cam.ac.uk>`_ and `LIBSVM <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_
| As far as possible, **SIDEKIT** as been made compatible with those tools by providing read and write
| functions in the appropriate formats and using similar structures.


ALIZE
-----

**SIDEKIT** is able to read and write in ALIZE binary format

   * a Gaussian Mixture Model
   * a label file
   * a matrix of statistics computed by using ``TotalVariability.exe`` or ``ComputeJFAStats.exe``.


BOSARIS
-------

| A part of the **BOSARIS** toolkit has been translated into Python
| in order to manipulate

   * enrollment lists as :ref:`IdMap` objects
   * trial lists as :ref:`Ndx` objects
   * score matrices as :ref:`Scores` objects
   * trial keys as :ref:`Key` objects

| to plot Detection Error Trade-off (DET) curves
| and compute minimum costs as defined by the `NIST <http://www.itl.nist.gov/iad/mig/tests/sre/>`_ . 

HTK
---

**SIDEKIT** is able to read and write in HTK format

   * a feature file (non-compressed)
   * a Gaussian Mixture Model (stored as a 3 states HMM)

LIBSVM
------

| **SIDEKIT** makes use of the LIBSVM library [Chang11]_ and its Python wrapper.
| High level interface are provided to train and test using SVMs.

