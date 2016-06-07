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
Copyright 2014-2016 Sylvain Meignier and Anthony Larcher

    :mod:`features_server` provides methods to manage features

"""
import numpy
import logging
import h5py

from sidekit.frontend.features import pca_dct, shifted_delta_cepstral, compute_delta
from sidekit.frontend.vad import label_fusion
from sidekit.frontend.normfeat import cms, cmvn, stg, cep_sliding_norm, rasta_filt
from sidekit.sv_utils import parse_mask


__license__ = "LGPL"
__author__ = "Anthony Larcher & Sylvain Meignier"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'

class FeaturesServer():
    """
    Classe qui ouvre un fichier HDF5
        - charge un ou plusieurs datasets
        - traite chaque dataset séparément
        - retourne une concaténation de l'ensemble
    """

    def __init__(self,
                 features_extractor=None,
                 feature_filename_structure=None,
                 sources=None,
                 dataset_list=None,
                 mask=None,
                 feat_norm=None,
                 vad=None,
                 dct_pca=False,
                 dct_pca_config=None,
                 sdc=False,
                 sdc_config=None,
                 delta=None,
                 double_delta=None,
                 delta_filter=None,
                 rasta=None,
                 double_channel_extension=None,
                 keep_all_features=None):
        """

        :param features_extractor: a FeaturesExtractor if required to extract features from audio file
        if None, data are loaded from an existing HDF5 file
        :param feature_filename_structure: structure of the filename to use to load HDF5 files
        :param sources: tuple of sources to load features different files
        :param dataset_list:
        :param mask:
        :param feat_norm:
        :param vad:
        :param dct_pca:
        :param dct_pca_config:
        :param sdc:
        :param sdc_config:
        :param delta:
        :param double_delta:
        :param delta_filter:
        :param rasta:
        :param double_channel_extension:
        :param keep_all_features:
        :return:
        """


        #:param features_extractor:
        #:param feature_filename_structure:
        #:param subservers:

        self.features_extractor = None
        self.feature_filename_structure = '{}'
        self.sources = ()
        self.dataset_list = None

        # Post processing options
        self.vad=None
        self.mask=None
        self.feat_norm=None
        self.dct_pca = False
        self.dct_pca_config = (12, 12, None)
        self.sdc=False
        self.sdc_config = (1, 3, 7)
        self.delta = False
        self.double_delta = False
        self.delta_filter = numpy.array([.25, .5, .25, 0, -.25, -.5, -.25])
        self.rasta=False
        self.double_channel_extension = ('_a', '_b')
        self.keep_all_features=True

        if features_extractor is not None:
            self.features_extractor = features_extractor
        if feature_filename_structure is not None:
            self.feature_filename_structure = feature_filename_structure
        if sources is not None:
            self.sources = sources
        if dataset_list is not None:
            self.dataset_list = dataset_list

        if vad is not None:
            self.vad = vad
        if mask is not None:
            self.mask = parse_mask(mask)
        if feat_norm is not None:
            self.feat_norm = feat_norm
        if dct_pca is not None:
            self.dct_pca = dct_pca
        if dct_pca_config is not None:
            self.dct_pca_config = dct_pca_config
        if sdc is not None:
            self.sdc = sdc
        if sdc_config is not None:
            self.sdc_config = sdc_config
        if delta is not None:
            self.delta = delta
        if double_delta is not None:
            self.double_delta = double_delta
        if delta_filter is not None:
            self.delta_filter = delta_filter
        if rasta is not None:
            self.rasta = rasta
        if double_channel_extension is not None:
            self.double_channel_extension = double_channel_extension
        if keep_all_features is not None:
            self.keep_all_features = keep_all_features

        self.show = 'empty'

    def __repr__(self):
        """

        :return: a string to display the object
        """
        ch = '\t show: {} \n\n'.format(self.show)
        ch += '\t feature_filename_structure: {} \n'.format(self.feature_filename_structure)
        ch += '\t  \n'
        ch += '\t  \n\n'
        ch += '\t Post processing options: \n'
        ch += '\t\t mask: {}  \n'.format(self.mask)
        ch += '\t\t feat_norm: {} \n'.format(self.feat_norm)
        ch += '\t\t vad: {} \n'.format(self.vad)
        ch += '\t\t dct_pca: {}, dct_pca_config: {} \n'.format(self.dct_pca,
                                                               self.dct_pca_config)
        ch += '\t\t sdc: {}, sdc_config: {} \n'.format(self.sdc,
                                                       self.sdc_config)
        ch += '\t\t delta: {}, double_delta: {}, delta_filter: {} \n'.format(self.delta,
                                                                             self.double_delta,
                                                                             self.delta_filter)
        ch += '\t\t rasta: {} \n'.format(self.rasta)
        ch += '\t\t keep_all_features: {} \n'.format(self.keep_all_features)

        return ch

    def post_processing(self, feat, label):
        """
        After cepstral coefficients or filter banks are computed or read from file
        post processing is applied
        :param cep:
        :param energy:
        :param label:
        :return:
        """

        # Apply a mask on the features
        if self.mask is not None:
            feat = self._mask(feat)

        # Perform RASTA filtering if required
        feat, label = self._rasta(feat, label)

        # Add temporal context
        if self.delta or self.double_delta:
            feat = self._delta_and_2delta(feat)
        elif self.dct_pca:
            feat = pca_dct(feat, self.dct_pca_config[0],
                          self.dct_pca_config[1],
                          self.dct_pca_config[2])
        elif self.sdc:
            feat = shifted_delta_cepstral(feat, d=self.sdc_config[0],
                                         P=self.sdc_config[1],
                                         k=self.sdc_config[2])

        # Smooth the labels and fuse the channels if more than one.
        logging.info('Smooth the labels and fuse the channels if more than one')
        if self.vad:
            label = label_fusion(label)


        # Normalize the data
        self._normalize(label, feat)

        # if not self.keep_all_features, only selected features and labels are kept
        if not self.keep_all_features:
            logging.info('no keep all')
            feat = feat[label]
            label = label[label]
        return feat, label

    def _mask(self, cep):
        """
        keep only the MFCC index present in the filter list
        :param cep:
        :return: return the list of MFCC given by filter list
        """
        if len(self.mask) == 0:
            raise Exception('filter list is empty')
        logging.debug('applied mask')
        return cep[:, self.mask]

    def _normalize(self, label, cep):
        """
        Normalize features in place

        :param label:
        :return:
        """
        # Perform feature normalization on the entire session.
        if self.feat_norm is None:
            logging.info('no norm')
            pass
        elif self.feat_norm == 'cms':
            logging.info('cms norm')
            cms(cep, label)
        elif self.feat_norm == 'cmvn':
            logging.info('cmvn norm')
            cmvn(cep, label)
        elif self.feat_norm == 'stg':
            logging.info('stg norm')
            stg(cep, label=label)
        elif self.feat_norm == 'cmvn_sliding':
            logging.info('sliding cmvn norm')
            cep_sliding_norm(cep, win=301, center=True, reduce=True)
        elif self.feat_norm == 'cms_sliding':
            logging.info('sliding cms norm')
            cep_sliding_norm(cep, win=301, center=True, reduce=False)
        else:
            logging.warning('Wrong feature normalisation type')

    def _delta_and_2delta(self, cep):
        """
        Add deltas and double deltas.
        :param cep: a matrix of cepstral cefficients

        :return: the cepstral coefficient stacked with deltas and double deltas
        """
        if self.delta:
            logging.info('add delta')
            delta = compute_delta(cep, filt=self.delta_filter)
            cep = numpy.column_stack((cep, delta))
            if self.double_delta:
                logging.info('add delta delta')
                double_delta = compute_delta(delta, filt=self.delta_filter)
                cep = numpy.column_stack((cep, double_delta))
        return cep

    def _rasta(self, cep, label):
        """
        Performs RASTA filtering if required.
        The two first frames are copied from the third to keep
        the length consistent
        !!! if vad is None: label[] is empty

        :param channel: name of the channel
        :return:
        """
        if self.rasta:
            logging.info('perform RASTA %s', self.rasta)
            cep = rasta_filt(cep)
            cep[:2, :] = cep[2, :]
            label[:2] = label[2]
        return cep, label

    def load(self, show, channel=0, input_feature_filename=None, label=None):
        """

        :param show:
        :param channel:
        :return:
        """
        """
        Si le nom du fichier d'entrée est totalement indépendant du show -> si feature_filename_structure ne contient pas "{}"
        on peut mettre à jour: self.audio_filename_structure pour entrer directement le nom du fichier de feature
        """
        feature_filename = None
        if input_feature_filename is not None:
            self.feature_filename_structure = input_feature_filename
            """
            On met à jour le feature_filename (que le show en fasse partie ou non)
            """
            feature_filename = self.feature_filename_structure.format(show)

        if self.dataset_list is not None:
            return self.get_features(show, channel=channel, input_feature_filename=feature_filename, label=label)
        else:
            logging.info('Extract tandem features from multiple sources')
            return self.get_tandem_features(show, channel=channel, label=label)

    def get_features(self, show, channel=0, input_feature_filename=None, label=None):
        """
        Get the datasets from a single HDF5 file
        The HDF5 file is loaded from disk or processed on the fly
        via the FeaturesExtractor of the current FeaturesServer
        :param h5f:
        :param show:
        :param channel:
        :param label_filename:
        :return:
        """

        #channel = 0
        #if show.endswith(self.double_channel_extension[1]):
        #    channel = 1

        """
        Si le nom du fichier d'entrée est totalement indépendant du show -> si feature_filename_structure ne contient pas "{}"
        on peut mettre à jour: self.audio_filename_structure pour entrer directement le nom du fichier de feature
        """
        if input_feature_filename is not None:
            self.feature_filename_structure = input_feature_filename

        # If no extractor for this source, open hdf5 file and return handler
        if self.features_extractor is None:
            h5f = h5py.File(self.feature_filename_structure.format(show))

        # If an extractor is provided for this source, extract features and return an hdf5 handler
        else:
            h5f = self.features_extractor.extract(show, channel, input_audio_filename=input_feature_filename)

        # Concatenate all required datasets
        feat = []
        if "energy" in self.dataset_list:
            feat.append(h5f.get("/".join((show, "energy"))).value[:, numpy.newaxis])
        if "cep" in self.dataset_list:
            feat.append(h5f.get("/".join((show, "cep"))).value)
        if "fb" in self.dataset_list:
            feat.append(h5f.get("/".join((show, "fb"))).value)
        if "bnf" in self.dataset_list:
            feat.append(h5f.get("/".join((show, "bnf"))).value)
        feat = numpy.hstack(feat)

        if label is None:
            if "/".join((show, "vad")) in h5f:
                label = h5f.get("/".join((show, "vad"))).value.astype('bool').squeeze()
            else:
                label = numpy.ones(feat.shape[0], dtype='bool')

        h5f.close()
        # Post-process the features and return the features and vad label
        return self.post_processing(feat, label)

    def get_tandem_features(self, show, channel=0, label=None):
        """

        :param show:
        :param feature_extractors:
        :return:
        """

        # Each source has its own sources (including subserver) that provides features and label
        features = []
        #label = numpy.empty(0)
        for features_server, get_vad in self.sources:

            # Get features from this source
            feat, lbl = features_server.get_features(show, channel=channel, label=label)

            if get_vad:
                label = lbl
            features.append(feat)

        features = numpy.hstack(features)

        # If the VAD is not required, return all labels at True
        if label is None:
            label = numpy.ones(feat.shape[0], dtype='bool')

        # Apply the final post-processing on the concatenated features
        return  self.post_processing(features, label)

    def mean_std(self, show, channel=0):
        feat, _ = self.load(show, channel=channel)[0][0]
        return feat.shape[0], feat.sum(axis=0), numpy.sum(feat**2, axis=0)
