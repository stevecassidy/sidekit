# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2016 Anthony Larcher and Sylvain Meignier

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""

PARALLEL_MODULE = 'multiprocessing'  # can be , threading, multiprocessing MPI is planned in the future
import numpy as np
param_type = np.float32
stat_type = np.float64

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
from sidekit.features_server_test import FeaturesServer_test
from sidekit.mixture import Mixture
from sidekit.statserver import StatServer

from sidekit.frontend.io import write_pcm
from sidekit.frontend.io import read_pcm
from sidekit.frontend.io import pcmu2lin
from sidekit.frontend.io import read_sph
from sidekit.frontend.io import write_label
from sidekit.frontend.io import read_label
from sidekit.frontend.io import read_spro4
from sidekit.frontend.io import read_audio
from sidekit.frontend.io import write_spro4
from sidekit.frontend.io import read_htk
from sidekit.frontend.io import write_htk

from sidekit.frontend.vad import vad_energy
from sidekit.frontend.vad import vad_snr
from sidekit.frontend.vad import label_fusion
from sidekit.frontend.vad import speech_enhancement


from sidekit.frontend.normfeat import cms
from sidekit.frontend.normfeat import cmvn
from sidekit.frontend.normfeat import stg
from sidekit.frontend.normfeat import rasta_filt


from sidekit.frontend.features import compute_delta
from sidekit.frontend.features import framing
from sidekit.frontend.features import pre_emphasis
from sidekit.frontend.features import trfbank
from sidekit.frontend.features import mel_filter_bank
from sidekit.frontend.features import mfcc
from sidekit.frontend.features import pca_dct
from sidekit.frontend.features import shifted_delta_cepstral

from sidekit.iv_scoring import cosine_scoring
from sidekit.iv_scoring import mahalanobis_scoring
from sidekit.iv_scoring import two_covariance_scoring
from sidekit.iv_scoring import PLDA_scoring

from sidekit.theano_utils import FForwardNetwork

from sidekit.sv_utils import clean_stat_server

__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2016 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'

# __all__ = ["io",
#            "vad",
#            "normfeat",
#            "features"
#            ]
