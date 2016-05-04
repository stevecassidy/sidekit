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
import os
import multiprocessing
import logging
from sidekit import PARALLEL_MODULE
from sidekit.frontend.features import *
from sidekit.frontend.vad import *
from sidekit.frontend.io import *
from sidekit.frontend.normfeat import *
from sidekit.sidekit_wrappers import *
import sys
import numpy as np
import ctypes

if sys.version_info.major == 3:
    import queue as Queue
else:
    import Queue
# import memory_profiler


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


class FeaturesManager:
    """
    A class for acoustic feature management.

    We assume that all features are stored in HDF5 format together with their VAD labels if they exits

    Can do:
        RASTA filtering
        1D or 2D DCT
        Feature normalization
        temporal_contextualization (deltas, PCA_DCT, SDC, add left and right context, TRAPS
    """

    def __init__(self, input_dir=None,
                 input_file_extension=None,
                 feature_id=None,
                 config=None,
                 sampling_frequency=None,
                 vad=None,
                 snr=None,
                 feat_norm=None,
                 log_e=None,
                 dct_pca=False,
                 dct_pca_config=None,
                 sdc=False,
                 sdc_config=None,
                 delta=None,
                 double_delta=None,
                 delta_filter=None,
                 rasta=None,
                 keep_all_features=None,
                 mask=None
                 ):
        """ Process of extracting the feature frames (LFCC or MFCC) from an audio signal.
        Speech Activity Detection, MFCC (or LFCC) extraction and normalization.
        Can include RASTA filtering, Short Term Gaussianization, MVN and delta
        computation.

        :param input_dir: directory where to find the audio files.
                Default is ./
        :param input_file_extension: extension of the audio files to read.
                Default is 'sph'.
        :param label_dir: directory where to store label files is required.
                Default is ./
        :param label_file_extension: extension of the label files to create.
                Default is '.lbl'.
        :param configuration file : 'diar_16k', 'sid_16k', 'diar_8k' or 'sid_8k'
        """

        self.input_dir = './'
        self.input_file_extension = '.wav'
        self.label_dir = './'
        self.label_file_extension = '.lbl'        
        self.from_file = 'audio'
        self.feature_id = 'ceps'
        self.single_channel_extension = [''],
        self.double_channel_extension = ['_a', '_b'],
        self.sampling_frequency = 8000
        self.lower_frequency = 0
        self.higher_frequency = self.sampling_frequency / 2.
        self.linear_filters = 0
        self.log_filters = 40
        self.window_size = 0.025
        self.shift = 0.01
        self.ceps_number = 13
        self.snr = 40
        self.vad = None
        self.feat_norm = None
        self.log_e = False
        self.dct_pca = False        
        self.dct_pca_config = (12, 12, None)
        self.sdc = False
        self.sdc_config = (1, 3, 7)
        self.delta = False
        self.double_delta = False
        self.delta_filter = np.array([.25, .5, .25, 0, -.25, -.5, -.25])
        self.mask = None
        self.rasta = False
        self.keep_all_features = False
        self.spec = False
        self.mspec = False

        # If a predefined config is chosen, apply it
        if config == 'diar_16k':
            self._config_diar_16k()
        elif config == 'diar_8k':
            self._config_diar_8k()
        elif config == 'sid_8k':
            self._config_sid_8k()
        elif config == 'sid_16k':
            self._config_sid_16k()
        elif config == 'fb_8k':
            self._config_fb_8k()
        elif config is None:
            pass
        else:
            raise Exception('unknown configuration value')

        # Manually entered parameters are applied
        if input_dir is not None:
            self.input_dir = input_dir
        if input_file_extension is not None:
            self.input_file_extension = input_file_extension
        if label_dir is not None:
            self.label_dir = label_dir
        if label_file_extension is not None:
            self.label_file_extension = label_file_extension
        if from_file is not None:
            self.from_file = from_file
        if feature_id is not None:
            self.feature_id = feature_id
        if single_channel_extension is not None:
            self.single_channel_extension = single_channel_extension
        if double_channel_extension is not None:
            self.double_channel_extension = double_channel_extension
        if sampling_frequency is not None:
            self.sampling_frequency = sampling_frequency
        if lower_frequency is not None:
            self.lower_frequency = lower_frequency
        if higher_frequency is not None:
            self.higher_frequency = higher_frequency
        if linear_filters is not None:
            self.linear_filters = linear_filters
        if log_filters is not None:
            self.log_filters = log_filters
        if window_size is not None:
            self.window_size = window_size
        if shift is not None:
            self.shift = shift
        if ceps_number is not None:
            self.ceps_number = ceps_number
        if snr is not None:
            self.snr = snr
        if vad is not None:
            self.vad = vad
        if feat_norm is not None:
            self.feat_norm = feat_norm
        if log_e is not None:
            self.log_e = log_e
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
        if mask is not None:
            self.mask = mask
        if rasta is not None:
            self.rasta = rasta
        if keep_all_features is not None:
            self.keep_all_features = keep_all_features
        if spec:
            self.spec = True
        if mspec:
            self.mspec = True
        
        self.cep = []
        self.label = []
        self.show = 'empty'
        self.audio_filename = 'empty'

    def __repr__(self):
        pass  # to do
        # ch = '\t show: {} keep_all_features: {} from_file: {}\n'.format(
        #     self.show, self.keep_all_features, self.from_file)
        # ch += '\t inputDir: {} inputFileExtension: {} \n'.format(self.input_dir,
        #                                                          self.input_file_extension)
        # ch += '\t labelDir: {}  labelFileExtension: {} \n'.format(
        #     self.label_dir, self.label_file_extension)
        # ch += '\t lower_frequency: {}  higher_frequency: {} \n'.format(
        #     self.lower_frequency, self.higher_frequency)
        # ch += '\t sampling_frequency: {} '.format(self.sampling_frequency)
        # ch += '\t linear_filters: {}  or log_filters: {} \n'.format(
        #     self.linear_filters, self.log_filters)
        # ch += '\t ceps_number: {}  window_size: {} shift: {} \n'.format(
        #     self.ceps_number, self.window_size, self.shift)
        # ch += '\t vad: {}  snr: {} \n'.format(self.vad, self.snr)
        # ch += '\t feat_norm: {} rasta: {} \n'.format(self.feat_norm, self.rasta)
        # ch += '\t log_e: {} delta: {} double_delta: {} \n'.format(self.log_e,
        #                                                           self.delta,
        #                                                           self.double_delta)
        #return ch


 
    def _config_to_define(self):
        """
        7 MFCC + 1 - 3 - 7 SDC
        """
        pass  # to do
        # self.sampling_frequency = 8000
        # self.lower_frequency = 300
        # self.higher_frequency = 3400
        # self.linear_filters = 0
        # self.log_filters = 24
        # self.window_size = 0.025
        # self.double_channel_extension = ['_a', '_b'],
        # self.shift = 0.01
        # self.ceps_number = 7
        # self.snr = 40
        # self.vad = 'snr'
        # self.feat_norm = None
        # self.log_e = False
        # self.delta = False
        # self.double_delta = False
        # self.sdc = True
        # self.sdc_config = (1, 3, 7)
        # self.rasta = False
        # self.keep_all_features = False

    def _vad(self, logEnergy, x, channel_ext, show):
        """
        Apply Voice Activity Detection.
        :param x:
        :param channel:
        :param window_sample:
        :param channel_ext:
        :param show:
        :return:
        """
        label = None
        if self.vad is None:
            logging.info('no vad')
            label = np.array([True] * logEnergy.shape[0])
        elif self.vad == 'snr':
            logging.info('vad : snr')
            window_sample = int(self.window_size * self.sampling_frequency)
            label = vad_snr(x, self.snr, fs=self.sampling_frequency,
                            shift=self.shift, nwin=window_sample)
        elif self.vad == 'energy':
            logging.info('vad : energy')
            label = vad_energy(logEnergy, distribNb=3,
                               nbTrainIt=8, flooring=0.0001,
                               ceiling=1.5, alpha=0.1)
        elif self.vad == 'lbl':  # load existing labels as reference
            logging.info('vad : lbl')
            for ext in channel_ext:
                label_filename = os.path.join(self.label_dir, show + ext + self.label_file_extension)
                label = read_label(label_filename)
        else:
            logging.warning('Wrong VAD type')
        return label

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

    def _delta_and_2delta(self, cep):
        """
        Add deltas and double deltas.
        :param cep: a matrix of cepstral cefficients
        
        :return: the cepstral coefficient stacked with deltas and double deltas
        """
        if self.delta:
            logging.info('add delta')
            delta = compute_delta(cep, filt=self.delta_filter)
            cep = np.column_stack((cep, delta))
            if self.double_delta:
                logging.info('add delta delta')
                double_delta = compute_delta(delta, filt=self.delta_filter)
                cep = np.column_stack((cep, double_delta))
        return cep

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
            for chan, c in enumerate(cep):
                cms(cep[chan], label[chan])
        elif self.feat_norm == 'cmvn':
            logging.info('cmvn norm')
            for chan, c in enumerate(cep):
                cmvn(cep[chan], label[chan])
        elif self.feat_norm == 'stg':
            logging.info('stg norm')
            for chan, c in enumerate(cep):
                stg(cep[chan], label=label[chan])
        elif self.feat_norm == 'cmvn_sliding':
            logging.info('sliding cmvn norm')
            for chan, c in enumerate(cep):
                cep_sliding_norm(cep[chan], win=301, center=True, reduce=True)
        elif self.feat_norm == 'cms_sliding':
            logging.info('sliding cms norm')
            for chan, c in enumerate(cep):
                cep_sliding_norm(cep[chan], win=301, center=True, reduce=False)
        else:
            logging.warning('Wrong feature normalisation type')

    def load(self, show, id=None):
        """
        Load acoustic coefficients from a session in HDF5, HTK or SPRO4 file.

        
        :param show: the name of the show to load
        :param id: the ID of the features to load (vad is reserved for for feature selection)
        
        :return: the array of selected features after possible processing (normalization, temporal_contextualization,
         RASTA filtering, 1D or 2D DCT, TRAPS
        """
        # test if features is already computed

        pass  # to modify

        # if self.show == show:
        #     return self.cep, self.label
        # self.show = show
        # if self.from_file == 'audio':
        #     logging.debug('compute MFCC: ' + show)
        #     logging.debug(self.__repr__())
        #     self.cep, self.label = self._features(show)
        # else:
        #     if self.from_file == 'pickle':
        #         logging.debug('load pickle: ' + show)
        #         input_filename = os.path.join(self.input_dir.format(s=show),
        #                                       show + self.input_file_extension)
        #         self.cep = [read_pickle(input_filename)]
        #     elif self.from_file == 'spro4':
        #         logging.debug('load spro4: ' + show)
        #         input_filename = os.path.join(self.input_dir.format(s=show),
        #                                       show + self.input_file_extension)
        #         self.cep = [read_spro4(input_filename)]
        #     elif self.from_file == 'htk':
        #         logging.debug('load htk: ' + show)
        #         input_filename = os.path.join(self.input_dir.format(s=show),
        #                                       show + self.input_file_extension)
        #         self.cep = [read_htk(input_filename)[0]]
        #     elif self.from_file == 'hdf5':
        #         logging.debug('load hdf5: ' + show)
        #         input_filename = os.path.join(self.input_dir +self.show + self.input_file_extension)
        #         with h5py.File(input_filename, "r") as hdf5_input_fh:
        #             cep, label = read_hdf5(hdf5_input_fh, show, feature_id=self.feature_id, vad=True)
        #             self.cep = [cep]
        #             self.label = [label]
        #             #self.cep = [read_cep_hdf5(hdf5_input_fh, show)]
        #     else:
        #         raise Exception('unknown from_file value')
        #
        #     # Load labels if needed
        #     if not self.from_file == 'hdf5':
        #         input_filename = os.path.join(self.label_dir.format(s=show), show + self.label_file_extension)
        #         if os.path.isfile(input_filename):
        #             self.label = [read_label(input_filename)]
        #             if self.label[0].shape[0] < self.cep[0].shape[0]:
        #                 missing = np.zeros(np.abs(self.cep[0].shape[0] - self.label[0].shape[0]), dtype='bool')
        #                 self.label[0] = np.hstack((self.label[0], missing))
        #         else:
        #             self.label = [np.array([True] * self.cep[0].shape[0])]
        #
        # if self.mask is not None:
        #     self.cep[0] = self._mask(self.cep[0])
        #     if len(self.cep) == 2:
        #         self.cep[1] = self._mask(self.cep[1])
        #
        # if not self.keep_all_features:
        #     logging.debug('!!! no keep all feature !!!')
        #     for chan in range(len(self.cep)):
        #         self.cep[chan] = self.cep[chan][self.label[chan]]
        #         self.label[chan] = self.label[chan][self.label[chan]]
        #
        # return self.cep, self.label

    def _mask(self, cep):
        """
        keep only the MFCC index present in the filter list
        :param cep:
        :return: return the list of MFCC given by filter list
        """
        pass   # to modify
    #    if len(self.mask) == 0:
    #        raise Exception('filter list is empty')
    #    logging.debug('applied mask')
    #    return cep[:, self.mask]

     def save(self, show, filename, mfcc_format, and_label=True):
         """
         Save the cep array in file

         :param show: the name of the show to save (loaded if need)
         :param filename: the file name of the mffc file or a list of 2 filenames
             for the case of double channel files
         :param mfcc_format: format of the mfcc file taken in values
             ['pickle', 'spro4', 'htk']
         :param and_label: boolean, if True save label files

         :raise: Exception if feature format is unknown
         """
        pass  # to modify
    #     self.load(show)
    #
    #     if len(self.cep) == 2:
    #         root, ext = os.path.splitext(filename)
    #         filename = [root + self.double_channel_extension[0] + ext,
    #                     root + self.double_channel_extension[1] + ext]
    #
    #     if mfcc_format.lower() == 'pickle':
    #         if len(self.cep) == 1 and self.cep[0].shape[0] > 0:
    #             logging.info('save pickle format: %s', filename)
    #             write_pickle(self.cep[0].astype(np.float32), filename)
    #         elif len(self.cep) == 2:
    #             logging.info('save pickle format: %s', filename[0])
    #             logging.info('save pickle format: %s', filename[1])
    #             if self.cep[0].shape[0] > 0:
    #                 write_pickle(self.cep[0].astype(np.float32), filename[0])
    #             if self.cep[1].shape[0] > 0:
    #                 write_pickle(self.cep[1].astype(np.float32), filename[1])
    #     elif mfcc_format.lower() == 'text':
    #         if len(self.cep) == 1 and self.cep[0].shape[0] > 0:
    #             logging.info('save text format: %s', filename)
    #             np.savetxt(filename, self.cep)
    #         elif len(self.cep) == 2:
    #             logging.info('save text format: %s', filename[0])
    #             logging.info('save text format: %s', filename[1])
    #             if self.cep[0].shape[0] > 0:
    #                 np.savetxt(filename[0], self.cep[0])
    #             if self.cep[1].shape[0] > 0:
    #                 np.savetxt(filename[1], self.cep[1])
    #     elif mfcc_format.lower() == 'spro4':
    #         if len(self.cep) == 1 and self.cep[0].shape[0] > 0:
    #             logging.info('save spro4 format: %s', filename)
    #             write_spro4(self.cep[0], filename)
    #         elif len(self.cep) == 2:
    #             logging.info('save spro4 format: %s', filename[0])
    #             logging.info('save spro4 format: %s', filename[1])
    #             if self.cep[0].shape[0] > 0:
    #                 write_spro4(self.cep[0], filename[0])
    #             if self.cep[1].shape[0] > 0:
    #                 write_spro4(self.cep[1], filename[1])
    #     elif mfcc_format.lower() == 'htk':
    #         if len(self.cep) == 1 and self.cep[0].shape[0] > 0:
    #             logging.info('save htk format: %s', filename)
    #             write_spro4(self.cep, filename)
    #         elif len(self.cep) == 2:
    #             logging.info('save htk format: %s', filename[0])
    #             logging.info('save htk format: %s', filename[1])
    #             if self.cep[0].shape[0] > 0:
    #                 write_htk(self.cep[0], filename[0])
    #             if self.cep[1].shape[0] > 0:
    #                 write_htk(self.cep[1], filename[1])
    #     elif self.from_file == 'hdf5':
    #         hdf5_ouput_fh = h5py.File(filename, "w")
    #         if len(self.cep) == 1 and self.cep[0].shape[0] > 0:
    #             logging.debug('save hdf5: ' + show)
    #             write_cep_hdf5(self.cep[0], hdf5_ouput_fh, show)
    #         elif len(self.cep) == 2:
    #             logging.info('save htk format: %s', show, self.double_channel_extension[0])
    #             logging.info('save htk format: %s', show, self.double_channel_extension[1])
    #             write_cep_hdf5(self.cep[0], hdf5_ouput_fh, show+'/'+self.double_channel_extension[0])
    #             write_cep_hdf5(self.cep[0], hdf5_ouput_fh, show+'/'+self.double_channel_extension[1])
    #         hdf5_ouput_fh.close()
    #     else:
    #         raise Exception('unknown feature format')
    #
    #     if and_label:
    #         if len(self.cep) == 1:
    #             output_filename = os.path.splitext(filename)[0] \
    #                                 + self.label_file_extension
    #             write_label(self.label[0], output_filename)
    #         elif len(self.cep) == 2:
    #             output_filename = [os.path.splitext(filename[0])[0] + self.label_file_extension,
    #                                os.path.splitext(filename[1])[0] + self.label_file_extension]
    #             write_label(self.label[0], output_filename[0])
    #             write_label(self.label[1], output_filename[1])

    @process_parallel_lists
    def save_list(self, audio_file_list, feature_file_list, mfcc_format, feature_dir, 
                  feature_file_extension, and_label=False, numThread=1):
        """
        Function that takes a list of audio files and extract features
        
        :param audio_file_list: an array of string containing the name of the feature 
            files to load
        :param feature_file_list: list of feature files to save, should correspond to the input audio_file_list
        :param mfcc_format: format of the feature files to save, could be spro4, htk, pickle
        :param feature_dir: directory where to save the feature files
        :param feature_file_extension: extension of the feature files to save
        :param and_label: boolean, if True save the label files
        :param numThread: number of parallel process to run
        """
        logging.info(self)
        for audio_file, feature_file in zip(audio_file_list, feature_file_list):
            cep_filename = os.path.join(feature_dir, feature_file + feature_file_extension)
            self.save(audio_file, cep_filename, mfcc_format, and_label)

    def dim(self):
        if self.show != 'empty':
            return self.cep[0].shape[1]
        dim = self.ceps_number
        if self.log_e:
            dim += 1
        if self.delta:
            dim *= 2
        if self.double_delta:
            dim *= 2
        logging.warning('cep dim computed using featureServer parameters')
        return dim

    def save_parallel(self, input_audio_list, output_feature_list, mfcc_format, feature_dir,
                      feature_file_extension, and_label=False, numThread=1):
        """
        Extract features from audio file using parallel computation
        
        :param input_audio_list: an array of string containing the name 
            of the audio files to process
        :param output_feature_list: an array of string containing the 
            name of the features files to save
        :param mfcc_format: format of the output feature files, could be spro4, htk, pickle
        :param feature_dir: directory where to save the feature files
        :param feature_file_extension: extension of the feature files to save
        :param and_label: boolean, if True save the label files
        :param numThread: number of parallel process to run
        """
        # Split the features to process for multi-threading
        loa = np.array_split(input_audio_list, numThread)
        lof = np.array_split(output_feature_list, numThread)
    
        jobs = []
        multiprocessing.freeze_support()
        for idx, feat in enumerate(loa):
            p = multiprocessing.Process(target=self.save_list,
                                        args=(loa[idx], lof[idx], mfcc_format, feature_dir,
                                              feature_file_extension, and_label))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

    def _load_and_stack_worker(self, input_queue, output):
        """Load a list of feature files into a Queue object
        
        :param input_queue: a Queue object
        :param output: a list of Queue objects to fill
        """
        while True:
            next_task = input_queue.get()
            
            if next_task is None:
                # Poison pill means shutdown
                output.put(None)
                input_queue.task_done()
                break
            
            # check which channel to keep from the file
            if next_task.endswith(self.double_channel_extension[0]) and (self.from_file == 'audio'):
                next_task = next_task[:-len(self.double_channel_extension[0])]
                output.put(self.load(next_task)[0][0])
            if next_task.endswith(self.double_channel_extension[1]) and self.from_file == 'audio':
                next_task = next_task[:-len(self.double_channel_extension[1])]
                output.put(self.load(next_task)[0][1])
            else:
                cep = self.load(next_task)[0][0]
                output.put(cep)
            
            input_queue.task_done()

    def load_and_stack(self, fileList, numThread=1):
        """Load a list of feature files and stack them in a unique ndarray. 
        The list of files to load is splited in sublists processed in parallel
        
        :param fileList: a list of files to load
        :param numThread: numbe of thead (optional, default is 1)
        """
        queue_in = multiprocessing.JoinableQueue(maxsize=len(fileList)+numThread)
        queue_out = []
        
        # Start worker processes
        jobs = []
        for i in range(numThread):
            queue_out.append(multiprocessing.Queue())
            p = multiprocessing.Process(target=self._load_and_stack_worker, 
                                        args=(queue_in, queue_out[i]))
            jobs.append(p)
            p.start()
        
        # Submit tasks
        for task in fileList:
            queue_in.put(task)

        for task in range(numThread):
            queue_in.put(None)
        
        # Wait for all the tasks to finish
        queue_in.join()
                   
        output = []
        for q in queue_out:
            while True:
                data = q.get()
                if data is None:
                    break
                output.append(data)

        for p in jobs:
            p.join()
        all_cep = np.concatenate(output, axis=0)

        return all_cep

    def load_and_stack_threading(self, fileList, numThread=1):
        """Load a list of feature files and stack them in a unique ndarray. 
        The list of files to load is splited in sublists processed in parallel
        
        :param fileList: a list of files to load
        :param numThread: numbe of thead (optional, default is 1)
        """
        queue_in = multiprocessing.JoinableQueue(maxsize=len(fileList)+numThread)
        queue_out = []
        
        # Start worker processes
        jobs = []
        for i in range(numThread):
            queue_out.append(Queue.Queue())
            p = threading.Thread(target=self._load_and_stack_worker, args=(queue_in, queue_out[i]))
            jobs.append(p)
            p.start()
        
        # Submit tasks
        for task in fileList:
            queue_in.put(task)

        for task in range(numThread):
            queue_in.put(None)
        
        # Wait for all the tasks to finish
        queue_in.join()
                   
        output = []
        for q in queue_out:
            while True:
                data = q.get()
                if data is None:
                    break
                output.append(data)

        for p in jobs:
            p.join()
        all_cep = np.concatenate(output, axis=0)

        return all_cep

    def mean_std(self, filename):
        feat = self.load(filename)[0][0]
        return feat.shape[0], feat.sum(axis=0), np.sum(feat**2, axis=0)
