# -*- coding: utf-8 -*-

# This package is a translation of a part of the BOSARIS toolkit.
# The authors thank Niko Brummer and Agnitio for allowing them to
# translate this code and provide the community with efficient structures
# and tools.
#
# The BOSARIS Toolkit is a collection of functions and classes in Matlab
# that can be used to calibrate, fuse and plot scores from speaker recognition
# (or other fields in which scores are used to test the hypothesis that two
# samples are from the same source) trials involving a model and a test segment.
# The toolkit was written at the BOSARIS2010 workshop which took place at the
# University of Technology in Brno, Czech Republic from 5 July to 6 August 2010.
# See the User Guide (available on the toolkit website)1 for a discussion of the
# theory behind the toolkit and descriptions of some of the algorithms used.
#
# The BOSARIS toolkit in MATLAB can be downloaded from `the website
# <https://sites.google.com/site/bosaristoolkit/>`_.

"""
This is the 'key' module
"""
import numpy as np
import os
import sys
import pickle
import gzip
import logging
from sidekit.bosaris.ndx import Ndx
from sidekit.sidekit_wrappers import check_path_existance
try:
    import h5py
    h5py_loaded = True
except ImportError:
    h5py_loaded = False


__author__ = "Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brummer", "Edward de Villiers"]



def diff(list1, list2):
    c = [item for item in list1 if item not in list2]
    c.sort()
    return c


def ismember(list1, list2):
    c = [item in list2 for item in list1]
    return c

class Key:
    """A class for representing a Key i.e. it classifies trials as                                                          
    target or non-target trials.

    :attr modelset: list of the models into a ndarray of strings
    :attr segset: list of the test segments into a ndarray of strings
    :attr tar: 2D ndarray of booleans which rows correspond to the models 
            and columns to the test segments. True if target trial.
    :attr non: 2D ndarray of booleans which rows correspond to the models 
            and columns to the test segments. True is non-target trial.
    """

    def __init__(self, keyFileName='', keyFileFormat='hdf5',
                models=np.array([]), testsegs=np.array([]), 
                trials=np.array([])):
        """Initialize a Key object.

        :param keyFileName: name of the file to load. Default is ''.
        :param keyFileFormat: format of the file to load. Can be:
            - 'pickle'
            - 'hdf5' (default)
            - 'txt'
        :param models: a list of models
        :param testsegs: a list of test segments
        :param trial: a list of trial types (target or nontarget)
        
        In case the keyFileName is empty, initialize an empty Key object.
        """
        self.modelset = np.empty(0, dtype="|O")
        self.segset = np.empty(0, dtype="|O")
        self.tar = np.array([], dtype="bool")
        self.non = np.array([], dtype="bool")

        if not h5py_loaded:
            keyFileFormat = 'pickle'

        if keyFileName == '':
            modelset = np.unique(models)
            segset = np.unique(testsegs)
    
            tar = np.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
            non = np.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
            
            
            for idx_m, model in enumerate(modelset):
                idx_current_model = np.argwhere(models == model).flatten()            
                current_model_keys = dict(zip(testsegs[idx_current_model], 
                                              trials[idx_current_model]))
                for idx_s, seg in enumerate(segset):
                    if seg in current_model_keys:
                        tar[idx_m, idx_s] = (current_model_keys[seg] == 'target')
                        non[idx_m, idx_s] = (current_model_keys[seg] == 'nontarget')
    
            self.modelset = modelset
            self.segset = segset
            self.tar = tar
            self.non = non
            assert self.validate(), "Wrong Key format"            
            
        elif keyFileFormat.lower() == 'pickle':
            self.read_pickle(keyFileName)
        elif keyFileFormat.lower() in ['hdf5', 'h5']:
            if h5py_loaded:
                self.read_hdf5(keyFileName)
            else:
                raise Exception('H5PY is not installed, chose another' +
                        ' format to load your Key')
        elif keyFileFormat.lower() == 'txt':
            self.read_txt(keyFileName)
        else:
            raise Exception('Wrong keyFileFormat')

    @check_path_existance
    def save(self, outputFileName):
        """Save the Key object to file. The format of the file
        to create is set accordingly to the extension of the filename.
        This extension can be '.txt' for text format
        and '.hdf5' or '.h5' for HDF5 format.

        :param outputFileName: name of the file to write to
        """
        extension = os.path.splitext(outputFileName)[1][1:].lower()
        if extension == 'p':
            self.save_pickle(outputFileName)
        elif extension in ['hdf5', 'h5']:
            if h5py_loaded:
                self.save_hdf5(outputFileName)
            else: 
                raise Exception('H5PY is not installed, chose another' +
                        ' format to save your Key')
        elif extension == 'txt':
            self.save_txt(outputFileName)
        else:
            raise Exception('Error: unknown extension')

    @check_path_existance
    def save_hdf5(self, outputFileName):
        """ Save Key in HDF5 format

        :param outputFileName: name of the file to write to
        """
        assert self.validate(), "Error: wrong Key format"

        with h5py.File(outputFileName, "w") as f:
            f.create_dataset("modelset", data=self.modelset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("segset", data=self.segset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            trialmask = np.array(self.tar, dtype='int8') - np.array(self.non, dtype='int8')
            f.create_dataset("trial_mask", data=trialmask,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)

    @check_path_existance
    def save_pickle(self, outputFileName):
        """Save Key in PICKLE format
        
        :param outputFileName: name of the file to write to
        """
        with gzip.open(outputFileName, "wb" ) as f:
            pickle.dump( self, f)

    @check_path_existance
    def save_txt(self, outputFileName):
        """Save a Key object to a text file.

        :param outputFileName: name of the output text file
        """
        fid = open(outputFileName, 'w')
        for m in range(self.modelset.shape[0]):
            segs = self.segset[self.tar[m, ]]
            for s in range(segs.shape[0]):
                fid.write('{} {} {}\n'.format(self.modelset[m],
                                                segs[s], 'target'))
            segs = self.segset[self.non[m, ]]
            for s in range(segs.shape[0]):
                fid.write('{} {} {}\n'.format(self.modelset[m],
                                                segs[s], 'nontarget'))
        fid.close()

    def filter(self, modlist, seglist, keep):
        """Removes some of the information in a key.  Useful for creating a
        gender specific key from a pooled gender key.  Depending on the
        value of \'keep\', the two input lists indicate the strings to
        retain or the strings to discard.

        :param modlist: a cell array of strings which will be compared with
            the modelset of 'inkey'.
        :param seglist: a cell array of strings which will be compared with
            the segset of 'inkey'.
        :param keep: a boolean indicating whether modlist and seglist are the
            models to keep or discard.

        :return: a filtered version of 'inkey'.
	    """
        if keep:
            keepmods = modlist
            keepsegs = seglist
        else:
            keepmods = diff(self.modelset, modlist)
            keepsegs = diff(self.segset, seglist)

        keepmodidx = np.array(ismember(self.modelset, keepmods))
        keepsegidx = np.array(ismember(self.segset, keepsegs))

        outkey = Key()
        outkey.modelset = self.modelset[keepmodidx]
        outkey.segset = self.segset[keepsegidx]
        tmp = self.tar[np.array(keepmodidx), :]
        outkey.tar = tmp[:, np.array(keepsegidx)]
        tmp = self.non[np.array(keepmodidx), :]
        outkey.non = tmp[:, np.array(keepsegidx)]

        assert(outkey.validate())

        if self.modelset.shape[0] > outkey.modelset.shape[0]:
            logging.info('Number of models reduced from %d to %d', self.modelset.shape[0],
                            outkey.modelset.shape[0])
        if self.segset.shape[0] > outkey.segset.shape[0]:
            logging.info('Number of test segments reduced from %d to %d',
                        self.segset.shape[0], outkey.segset.shape[0])
        return outkey

    def to_ndx(self):
        """Create a Ndx object based on the Key object

        :return: a Ndx object based on the Key
        """
        ndx = Ndx()
        ndx.modelset = self.modelset
        ndx.segset = self.segset
        ndx.trialmask = self.tar | self.non
        return ndx

    def validate(self):
        """Checks that an object of type Key obeys certain rules that
        must always be true.

        :return: a boolean value indicating whether the object is valid.
        """
        ok = isinstance(self.modelset, np.ndarray)
        ok &= isinstance(self.segset, np.ndarray)
        ok &= isinstance(self.tar, np.ndarray)
        ok &= isinstance(self.non, np.ndarray)
        ok &= self.modelset.ndim == 1
        ok &= self.segset.ndim == 1
        ok &= self.tar.ndim == 2
        ok &= self.non.ndim == 2
        ok &= self.tar.shape == self.non.shape
        ok &= self.tar.shape[0] == self.modelset.shape[0]
        ok &= self.tar.shape[1] == self.segset.shape[0]
        return ok

    def read(self, inputFileName):
        """Reads information from a file and constructs a Key object.  
        The type of file is deduced from the extension.

        :param inputFileName: name of the file to read from
        """
        extension = os.path.splitext(inputFileName)[1][1:].lower()
        if extension == 'p':
            self.read_pickle(inputFileName)
        elif extension in ['hdf5', 'h5']:
            if h5py_loaded:
                read_key_hdf5(self, inputFileName)
            else: 
                raise Exception('H5PY is not installed, chose another' +
                        ' format to load your Key')
        elif extension == 'txt':
            self.read_txt(inputFileName)
        else:
            raise Exception('Error: unknown extension')

    def read_hdf5(self, inputFileName):
        """Reads a Key object from an hdf5 file.
  
        :param inputFileName: name of the file to read from
        """
        with h5py.File(inputFileName, "r") as f:

            self.modelset = f.get("modelset").value
            self.segset = f.get("segset").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                self.modelset = self.modelset.astype('U100', copy=False)
                self.segset = self.segset.astype('U100', copy=False)

            trialmask = f.get("trial_mask").value
            self.tar = (trialmask == 1)
            self.non = (trialmask == -1)

            assert self.validate(), "Error: wrong Key format"

    def read_pickle(self, inputFileName):
        """Read Key in PICKLE format.
        
        :param inputFileName: name of the file to read from
        """        
        with gzip.open(inputFileName,'rb') as f:
            key = pickle.load(f)
            self.non = key.non
            self.tar = key.tar
            self.modelset = key.modelset
            self.segset = key.segset

    def read_txt(self, inputFileName):
        """Creates a Key object from information stored in a text file.

	    :param inputFileName: name of the file to read from
        """
        models, testsegs, trial  = np.loadtxt(inputFileName, delimiter=' ', 
                                        dtype={'names': ('mod', 'seg', 'key'), 
                                        'formats': ('S1000', 'S1000', 'S10')},
                                        unpack=True)

        models = models.astype('|O', copy=False).astype('S', copy=False)
        testsegs = testsegs.astype('|O', copy=False).astype('S', copy=False)
        trial = trial.astype('|O', copy=False).astype('S', copy=False)

        if sys.version_info[0] == 3:
            models = models.astype('U', copy=False)
            testsegs = testsegs.astype('U', copy=False)
            trial = trial.astype('U', copy=False)

        modelset = np.unique(models)
        segset = np.unique(testsegs)

        tar = np.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
        non = np.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
        
        
        for idx_m, model in enumerate(modelset):
            idx_current_model = np.argwhere(models == model).flatten()            
            current_model_keys = dict(zip(testsegs[idx_current_model], 
                                          trial[idx_current_model]))
            for idx_s, seg in enumerate(segset):
                if seg in current_model_keys:
                    tar[idx_m, idx_s] = (current_model_keys[seg] == 'target')
                    non[idx_m, idx_s] = (current_model_keys[seg] == 'nontarget')

        self.modelset = modelset
        self.segset = segset
        self.tar = tar
        self.non = non
        assert self.validate(), "Wrong Key format"

    def merge(self, keyList):
        """Merges Key objects. This function takes as input a list of
        Key objects to merge in the curent one.

        :param keyList: the list of Keys to merge
        """
        # the output key must have all models and segment in the input
        # keys (only once) and the same target and non-target trials.
        # It is an error if a trial is a target in one key and a
        # non-target in another, but a target or non-target marker will
        # override a 'non-trial' marker.
        assert isinstance(keyList, list), "Input is not a list"
        for key in keyList:
            assert isinstance(keyList, list), \
                    '{} {} {}'.format("Element ", key, " is not a list")

        for key2 in keyList:
            key_new = Key()
            key1 = self

            # create new ndx with empty masks
            key_new.modelset = np.union1d(key1.modelset, key2.modelset)
            key_new.segset = np.union1d(key1.segset, key2.segset)

            # expand ndx1 mask
            tar_1 = np.zeros((key_new.modelset.shape[0],
                                key_new.segset.shape[0]),
                                dtype="bool")
            non_1 = np.zeros((key_new.modelset.shape[0],
                                key_new.segset.shape[0]), dtype="bool")
            model_index_a = np.argwhere(np.in1d(key_new.modelset,
                                key1.modelset))
            model_index_b = np.argwhere(np.in1d(key1.modelset,
                                key_new.modelset))
            seg_index_a = np.argwhere(np.in1d(key_new.segset, key1.segset))
            seg_index_b = np.argwhere(np.in1d(key1.segset, key_new.segset))
            tar_1[model_index_a[:, None], seg_index_a] \
                    = key1.tar[model_index_b[:, None], seg_index_b]
            non_1[model_index_a[:, None], seg_index_a] \
                    = key1.non[model_index_b[:, None], seg_index_b]

            # expand ndx2 mask
            tar_2 = np.zeros((key_new.modelset.shape[0],
                                key_new.segset.shape[0]), dtype="bool")
            non_2 = np.zeros((key_new.modelset.shape[0],
                                key_new.segset.shape[0]), dtype="bool")
            model_index_a = np.argwhere(np.in1d(key_new.modelset,
                                                key2.modelset))
            model_index_b = np.argwhere(np.in1d(key2.modelset,
                                                key_new.modelset))
            seg_index_a = np.argwhere(np.in1d(key_new.segset, key2.segset))
            seg_index_b = np.argwhere(np.in1d(key2.segset, key_new.segset))
            tar_2[model_index_a[:, None], seg_index_a] \
                    = key2.tar[model_index_b[:, None], seg_index_b]
            non_2[model_index_a[:, None], seg_index_a] \
                    = key2.non[model_index_b[:, None], seg_index_b]

            # merge masks
            tar = tar_1 | tar_2
            non = non_1 | non_2

            # check for clashes
            assert np.sum(tar & non) == 0, "Conflict in the new Key"

            # build new key
            key_new.tar = tar
            key_new.non = non
            self.modelset = key_new.modelset
            self.segset = key_new.segset
            self.tar = key_new.tar
            self.non = key_new.non
            self.validate()

