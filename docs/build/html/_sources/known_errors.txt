Known errors and warnings
=========================

Warning due to the ctypes for multiprocessing...::

   ctypeslib.py:408: RuntimeWarning: Item size computed from the PEP 3118 buffer format string does not match the actual item size.

Pickle files created with version of Python below 2 and 3 might not be readable with Python 3 and 2.

It might happen that the PLDA does not converge, especially after `i`-vector normalization. This is due to the version of Lapack and Blas available on the
machine and used by scipy. To solve this issue, install OpenBlas and Lapack and link to scipy. We srongly recommand not to use ALTAS.
