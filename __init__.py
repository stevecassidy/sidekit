# -*- coding: utf-8 -*-
"""
Copyright 2014-2016 Anthony Larcher

.. topic::sidekit

|    This file is part of SIDEKIT.
|
|    SIDEKIT is a python package for speaker verification.
|    Home page: http://www-lium.univ-lemans.fr/sidekit/
|    
|    SIDEKIT is free software: you can redistribute it and/or modify
|    it under the terms of the GNU Lesser General Public License as 
|    published by the Free Software Foundation, either version 3 of the License, 
|    or (at your option) any later version.
|
|    SIDEKIT is distributed in the hope that it will be useful,
|    but WITHOUT ANY WARRANTY; without even the implied warranty of
|    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
|    GNU Lesser General Public License for more details.
|
|    You should have received a copy of the GNU Lesser General Public License
|    along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""

PARALLEL_MODULE = 'multiprocessing'  # can be , threading, multiprocessing MPI is planned in the future

import sys

# Import libsvm
import logging
from ctypes import *
from ctypes.util import find_library
from os import path

from sidekit.sidekit_wrappers import *

# Import bosaris-like classes
from sidekit.bosaris import IdMap
from sidekit.bosaris import Ndx
from sidekit.bosaris import Key
from sidekit.bosaris import Scores
from sidekit.bosaris import DetPlot
from sidekit.bosaris import effective_prior
from sidekit.bosaris import fast_minDCF

# Import classes
from sidekit.features_server import FeaturesServer
from sidekit.mixture import Mixture
from sidekit.statserver import StatServer

import sidekit.frontend.io
import sidekit.frontend.vad
import sidekit.frontend.normfeat
import sidekit.frontend.features

# Import function libraries
from sidekit.sidekit_io import *
from sidekit.sv_utils import *
from sidekit.lid_utils import *
from sidekit.gmm_scoring import *
from sidekit.jfa_scoring import *
from sidekit.iv_scoring import *

from sidekit.theano_utils import *

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__version__ = "1.0.4"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


libsvm_loaded = False
try:
    dirname = os.path.join(path.dirname(path.abspath(__file__)), 'libsvm')
    if sys.platform == 'win32':
        libsvm = CDLL(path.join(dirname, r'libsvm.dll'))
        libsvm_loaded = True
    else:
        libsvm = CDLL(path.join(dirname, 'libsvm.so.2'))
        libsvm_loaded = True
except:
    # For unix the prefix 'lib' is not considered.
    if find_library('svm'):
        libsvm = CDLL(find_library('svm'))
        libsvm_loaded = True
    elif find_library('libsvm'):
        libsvm = CDLL(find_library('libsvm'))
        libsvm_loaded = True
    else:
        libsvm_loaded = False
        logging.warning('WARNNG: libsvm is not installed, please refer to the' +
                        ' documentation if you intend to use SVM classifiers')

if libsvm_loaded:
    from sidekit.libsvm import *
    from sidekit.svm_scoring import *
    from sidekit.svm_training import *

__all__ = ["bosaris",
           "frontend",
           "libsvm",
           "frontend",
           "sv_utils",
           "gmm_scoring",
           "svm_scoring",
           "svm_training",
           "iv_scoring",
           "sidekit_io",
           "mixture",
           "statserver",
           "features_server",
	   "theano_utils"]
