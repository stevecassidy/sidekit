import numpy
import h5py
import logging
from sidekit import PARAM_TYPE
from sidekit.frontend.features import mfcc
from sidekit.frontend.io import read_audio, read_label, write_hdf5
from sidekit.frontend.vad import vad_snr, vad_energy
from sidekit.sidekit_wrappers import process_parallel_lists

class FeaturesExtractor():

    """
    Charge un fichier audio (SPH, WAVE, RAW PCM)
    Extrait 1 unique canal
    Retourne un tuple contenant:
        (VAD, FB, CEPS, BNF)
         selon les options choisies
    """

    def __init__(self,
                 audio_filename_structure=None,
                 feature_filename_structure=None,
                 sampling_frequency=None,
                 lower_frequency=None,
                 higher_frequency=None,
                 filter_bank=None,
                 filter_bank_size=None,
                 window_size=None,
                 shift=None,
                 ceps_number=None,
                 vad=None,
                 snr=None,
                 pre_emphasis=None,
                 save_param=None,
                 keep_all_features=None,
                 single_channel_extension=None,
                 double_channel_extension=None):
        """
        :param show: name of the file that will be used in the filename_structure
        :param filename_structure: string to format to include the show
        :param sampling_frequency: optional, if processing RAW PCM
        :param filter_bank: type of fiter scale to use, can be lin or log (for linear of log-scale)
        :param filter_bank_size: number of filters bands
        :param save_param: tuple of 5 boolean, if True then save to file the parameters in the following order:
                (cep, energy, fb, bnf, vad_label)
                For instance: save_param=(True, False, True, False, False) will save to disk the cepstral coefficients
                and filter-banks only
        :return:
        """

        # Set the default values
        self.audio_filename_structure = None
        self.feature_filename_structure = '{}'
        self.sampling_frequency = 8000
        self.lower_frequency = None
        self.higher_frequency = None
        self.filter_bank = None
        self.filter_bank_size = None
        self.window_size = None
        self.shift = None
        self.ceps_number = None
        self.vad = None
        self.snr = None
        self.pre_emphasis = 0.97
        self.save_param=(True, True, True, True, True)
        self.keep_all_features = None
        self.single_channel_extension = ('')
        self.double_channel_extension = ('_a', '_b')


        if audio_filename_structure is not None:
            self.audio_filename_structure = audio_filename_structure
        if feature_filename_structure is not None:
            self.feature_filename_structure = feature_filename_structure
        if sampling_frequency is not None:
            self.sampling_frequency = sampling_frequency
        if lower_frequency is not None:
            self.lower_frequency = lower_frequency
        if higher_frequency is not None:
            self.higher_frequency = higher_frequency
        if filter_bank is not None:
            self.filter_bank = filter_bank
        if filter_bank_size is not None:
            self.filter_bank_size = filter_bank_size
        if window_size is not None:
            self.window_size = window_size
        if shift is not None:
            self.shift = shift
        if ceps_number is not None:
            self.ceps_number = ceps_number
        if vad is not None:
            self.vad = vad
        if snr is not None:
            self.snr = snr
        if pre_emphasis is not None:
            self.pre_emphasis = pre_emphasis
        if save_param is not None:
            self.save_param = save_param
        if keep_all_features is not None:
            self.keep_all_features = keep_all_features
        if single_channel_extension is not None:
            self.single_channel_extension = single_channel_extension
        if double_channel_extension is not None:
            self.double_channel_extension = double_channel_extension

        self.window_sample = None
        if not (self.window_size is None or self.sampling_frequency is None):
            self.window_sample = int(self.window_size * self.sampling_frequency)

        self.shift_sample = None
        if not (self.shift is None or self.sampling_frequency is None):
            self.shift_sample = int(self.shift * self.sampling_frequency)

        self.show = 'empty'

    def __repr__(self):
        ch = '\t show: {} keep_all_features: {}\n'.format(
            self.show, self.keep_all_features)
        ch += '\t audio_filename_structure: {}  \n'.format(self.audio_filename_structure)
        ch += '\t feature_filename_structure: {}  \n'.format(self.feature_filename_structure)
        ch += '\t pre-emphasis: {} \n'.format(self.pre_emphasis)
        ch += '\t lower_frequency: {}  higher_frequency: {} \n'.format(
            self.lower_frequency, self.higher_frequency)
        ch += '\t sampling_frequency: {} \n'.format(self.sampling_frequency)
        ch += '\t filter bank: {} filters of type {}\n'.format(
            self.filter_bank_size, self.filter_bank)
        ch += '\t ceps_number: {} \n\t window_size: {} shift: {} \n'.format(
            self.ceps_number, self.window_size, self.shift)
        ch += '\t vad: {}  snr: {} \n'.format(self.vad, self.snr)
        ch += '\t single channel extension: {} \n'.format(self.single_channel_extension)
        ch += '\t double channel extension: {} \n'.format(self.double_channel_extension)
        return ch


    def extract(self, show, channel, input_audio_filename=None, output_feature_filename=None, backing_store=False):
        """

        :return:
        """
        # Create the filename to load
        """
        Si le nom du fichier d'entrée est totalement indépendant du show -> si audio_filename_structure ne contient pas "{}"
        on peut mettre à jour: self.audio_filename_structure pour entrer directement le nom du fichier audio
        """
        if input_audio_filename is not None:
            self.audio_filename_structure = input_audio_filename
        """
        On met à jour l'audio_filename (que le show en fasse partie ou non)
        """
        audio_filename = self.audio_filename_structure.format(show)

        """
        Si le nom du fichier de sortie est totalement indépendant du show -> si feature_filename_structure ne contient pas "{}"
        on peut mettre à jour: self.audio_filename_structure pour entrer directement le nom du fichier de feature
        """
        #if (not '{}' in self.feature_filename_structure) and output_feature_filename is not None:
        if output_feature_filename is not None:
            self.feature_filename_structure = output_feature_filename
        """
        On met à jour le feature_filename (que le show en fasse partie ou non)
        """
        feature_filename = self.feature_filename_structure.format(show)

        # Open audio file, get the signal and possibly the sampling frequency
        signal, sample_rate = read_audio(audio_filename, self.sampling_frequency)
        if signal.ndim == 1:
            signal = signal[:, numpy.newaxis]

        # Process the target channel to return Filter-Banks, Cepstral coefficients and BNF if required
        length, chan = signal.shape

        # If the size of the signal is not enough for one frame, return zero features
        if length < self.window_sample:
            cep   = numpy.empty((0, self.ceps_number), dtype=PARAM_TYPE)
            energy = numpy.empty((0, 1), dtype=PARAM_TYPE)
            fb    = numpy.empty((0, self.filter_bank_size), dtype=PARAM_TYPE)
            label = numpy.empty((0, 1), dtype='int8')

        else:
            # Random noise is added to the input signal to avoid zero frames.
            numpy.random.seed(0)
            signal[:, channel] += 0.0001 * numpy.random.randn(signal.shape[0])

            dec = self.shift_sample * 250 * 25000 + self.window_sample
            dec2 = self.window_sample - self.shift_sample
            start = 0
            end = min(dec, length)

            # Process the signal by batch to avoid problems for very long signals
            while start < (length - dec2):
                logging.info('process part : %f %f %f',
                             start / self.sampling_frequency,
                             end / self.sampling_frequency,
                             length / self.sampling_frequency)

                # Extract cepstral coefficients, energy and filter banks
                cep, energy, _, fb = mfcc(signal[start:end, channel],
                         fs=self.sampling_frequency,
                         lowfreq=self.lower_frequency,
                         maxfreq=self.higher_frequency,
                         nlinfilt=self.filter_bank_size if self.filter_bank == "lin" else 0,
                         nlogfilt=self.filter_bank_size if self.filter_bank == "log" else 0,
                         nwin=self.window_size,
                         nceps=self.ceps_number,
                         get_spec=False,
                         get_mspec=True,
                         prefac=self.pre_emphasis)

                # Perform feature selection
                label = self._vad(cep, energy, fb, signal[start:end, channel])

                start = end - dec2
                end = min(end + dec, length)
                if cep.shape[0] > 0:
                    logging.info('!! size of signal cep: %f len %d type size %d', cep[-1].nbytes/1024/1024, len(cep[-1]),
                             cep[-1].nbytes/len(cep[-1]))

        # Create the HDF5 file
        h5f = h5py.File(feature_filename, 'a', backing_store=backing_store, driver='core')
        if not self.save_param[0]:
            cep = None
        if not self.save_param[1]:
            energy = None
        if not self.save_param[2]:
            fb = None
        if not self.save_param[3]:
            bnf = None
        if not self.save_param[4]:
            label = None
        write_hdf5(show, h5f, cep, energy, fb, None, label)

        return h5f

    def save(self, show, channel, input_audio_filename=None, output_feature_filename=None):
        """
        TO DO: BNF are not yet managed here
        :param show:
        :param channel:
        """
        # Load the cepstral coefficients, energy, filter-banks, bnf and vad labels
        h5f = self.extract(show, channel, input_audio_filename, output_feature_filename, backing_store=True)

        # Write the hdf5 file to disk
        h5f.close()

    def _vad(self, cep, logEnergy, fb, x, label_filename=None):
        """
        Apply Voice Activity Detection.
        :param cep:
        :param logEnergy:
        :param fb:
        :param x:
        :return:
        """
        label = None
        if self.vad is None:
            logging.info('no vad')
            label = numpy.array([True] * logEnergy.shape[0])
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
        elif self.vad == 'dnn':
            pass  # TO DO
        elif self.vad == 'lbl':  # load existing labels as reference
            logging.info('vad : lbl')
            label = read_label(label_filename)
        else:
            logging.warning('Wrong VAD type')
        return label

    @process_parallel_lists
    def save_list(self,
                  show_list,
                  channel_list,
                  audio_file_list=None,
                  feature_file_list=None,
                  numThread=1):
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
        for show, channel, audio_file, feature_file in zip(show_list, channel_list, audio_file_list, feature_file_list):
            self.save(show, channel, audio_file, feature_file)

