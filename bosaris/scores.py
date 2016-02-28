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
This is the 'scores' module

"""
import numpy as np
import os
import pickle
import gzip
import logging
from sidekit.bosaris.ndx import Ndx
from sidekit.bosaris.key import Key
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


class Scores:
    """A class for storing scores for trials.  The modelset and segset
    fields are lists of model and test segment names respectively.
    The element i,j of scoremat and scoremask corresponds to the
    trial involving model i and test segment j.

    :attr modelset: list of unique models in a ndarray 
    :attr segset: list of unique test segments in a ndarray
    :attr scoremask: 2D ndarray of boolean which indicates the trials of interest 
            i.e. the entry i,j in scoremat should be ignored if scoremask[i,j] is False
    :attr scoremat: 2D ndarray of scores
    """

    def __init__(self, scoresFileName='', scoresFileFormat='hdf5'):
        """ Initialize a Scores object by loading information from a file
        in PICKLE, HDF5 of text format.

        :param scoresFileName: name of the file to load
        :param scoresFileFormat: format of the file to load. Can be:
            - 'pickle' for PICKLE format
            - 'hdf5' for HDF5 format
            - 'txt' for text format

        Default is 'hdf5', if h5py is not imported, pickle format is used
        """
        self.modelset = np.empty(0, dtype="|O")
        self.segset = np.empty(0, dtype="|O")
        self.scoremask = np.array([], dtype="bool")
        self.scoremat = np.array([])

        if not h5py_loaded:
            scoresFileFormat = 'pickle'

        if scoresFileName == '':
            pass
        elif scoresFileFormat == 'pickle':
            self.read_pickle(scoresFileName)
        elif scoresFileFormat == 'hdf5':
            if h5py_loaded:
                self.read_hdf5(scoresFileName)
            else:
                raise Exception('h5py is not installed, chose another' +
                        ' format to load your Scores')
        elif scoresFileFormat == 'txt':
            self.read_txt(scoresFileName)
        else:
            raise Exception('Wrong scoresFileFormat')

    @check_path_existance
    def save(self, outputFileName):
        """Save the Scores object to file. The format of the file is deduced from
        the extension of the filename. The format can be PICKLE, HDF5 or text.
        Extension for text file should be '.p' for pickle '.txt' 
        and for HDF5 it should be '.hdf5' or '.h5'

        :param outputFileName: name of the file to write to
        """
        extension = os.path.splitext(outputFileName)[1][1:].lower()
        if extension == 'p':
            self.save_pickle(outputFileName)
        elif extension in ['hdf5', 'h5']:
            if h5py_loaded:
                self.save_hdf5(outputFileName)
            else:
                raise Exception('h5py is not installed, chose another' +
                        ' format to save your Scores')
        elif extension == 'txt':
            self.save_txt(outputFileName)
        else:
            raise Exception('Error: unknown extension')

    @check_path_existance
    def save_hdf5(self, outputFileName):
        """ Save Scores in HDF5 format

        :param outputFileName: name of the file to write to
        """
        with h5py.File(outputFileName, "w") as f:
            f.create_dataset("modelset", data=self.modelset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("segest", data=self.segset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("score_mask", data=self.scoremask.astype('int8'),
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("scores", data=self.scoremat,
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)

    @check_path_existance
    def save_pickle(self, outputFileName):
        """Save Scores in PICKLE format. If Python > 3.3, scores are converted
        to float32 before saving to save space.
        
        :param outputFileName: name of the file to write to
        """
        with gzip.open(outputFileName, "wb" ) as f:
            self.scoremat.astype('float32', copy=False)
            pickle.dump( self, f)

    @check_path_existance
    def save_txt(self, outputFileName):
        """Save a Scores object in a text file
	
        :param outputFileName: name of the file to write to
        """
        if not os.path.exists(os.path.dirname(outputFileName)):
            os.makedirs(os.path.dirname(outputFileName))
        
        with open(outputFileName, 'w') as fid:
            for m in range(self.modelset.shape[0]):
                segs = self.segset[self.scoremask[m, ]]
                scores = self.scoremat[m, self.scoremask[m, ]]
                for s in range(segs.shape[0]):
                    fid.write('{} {} {}\n'.format(self.modelset[m],
                                                segs[s], scores[s]))

    def get_tar_non(self, key):
        """Divides scores into target and non-target scores using
        information in a key.

        :param key: a Key object.

        :return: a vector of target scores.
            :return: a vector of non-target scores.
        """
        newScore = self.align_with_ndx(key)
        tarndx = key.tar & newScore.scoremask
        nonndx = key.non & newScore.scoremask
        tar = newScore.scoremat[tarndx]
        non = newScore.scoremat[nonndx]
        return tar, non

    def align_with_ndx(self, ndx):
        """The ordering in the output Scores object corresponds to ndx, so
        aligning several Scores objects with the same ndx will result in
        them being comparable with each other.

        :param ndx: a Key or Ndx object

        :return: resized version of the current Scores object to size of \'ndx\'
                and reordered according to the ordering of modelset and segset in \'ndx\'.
        """
        aligned_scr = Scores()
        aligned_scr.modelset = ndx.modelset
        aligned_scr.segset = ndx.segset

        hasmodel = np.array(ismember(ndx.modelset, self.modelset))
        rindx = np.array([np.argwhere(self.modelset == v)[0][0]
                            for v in ndx.modelset[hasmodel]])
        hasseg = np.array(ismember(ndx.segset, self.segset))
        cindx = np.array([np.argwhere(self.segset == v)[0][0]
                            for v in ndx.segset[hasseg]])

        aligned_scr.scoremat = np.zeros((ndx.modelset.shape[0],
                                         ndx.segset.shape[0]))
        aligned_scr.scoremat[np.where(hasmodel)[0][:, None],
                np.where(hasseg)[0]] = self.scoremat[rindx[:, None], cindx]

        aligned_scr.scoremask = np.zeros((ndx.modelset.shape[0],
                                        ndx.segset.shape[0]), dtype='bool')
        aligned_scr.scoremask[np.where(hasmodel)[0][:, None],
            np.where(hasseg)[0]] = self.scoremask[rindx[:, None], cindx]

        assert np.sum(aligned_scr.scoremask) \
                <= (np.sum(hasmodel) * np.sum(hasseg)), 'Error in new scoremask'

        if isinstance(ndx, Ndx):
            aligned_scr.scoremask = aligned_scr.scoremask & ndx.trialmask
        else:
            aligned_scr.scoremask = aligned_scr.scoremask & (ndx.tar | ndx.non)

        if np.sum(hasmodel) < ndx.modelset.shape[0]:
            logging.info('models reduced from %d to %d', ndx.modelset.shape[0],
                        np.sum(hasmodel))
        if np.sum(hasseg) < ndx.segset.shape[0]:
            logging.info('testsegs reduced from %d to %d', ndx.segset.shape[0],
                        np.sum(hasseg))

        if isinstance(ndx, Key):
            tar = ndx.tar & aligned_scr.scoremask
            non = ndx.non & aligned_scr.scoremask

            missing = np.sum(ndx.tar) - np.sum(tar)
            if missing > 0:
                logging.info('%d of %d targets missing', missing, np.sum(ndx.tar))
            missing = np.sum(ndx.non) - np.sum(non)
            if missing > 0:
                logging.info('%d of %d non targets missing', missing, np.sum(ndx.non))

        else:
            mask = ndx.trialmask & aligned_scr.scoremask
            missing = np.sum(ndx.trialmask) - np.sum(mask)
            if missing > 0:
                logging.info('%d of %d trials missing', missing, np.sum(ndx.trialmask))

        assert all(np.isfinite(aligned_scr.scoremat[aligned_scr.scoremask])), \
                'Inifinite or Nan value in the scoremat'
        assert aligned_scr.validate(), 'Wrong Score format'
        return aligned_scr

    def set_missing_to_value(self, ndx, value):
        """Sets all scores for which the trialmask is true but the scoremask
        is false to the same value, supplied by the user.

        :param ndx: a Key or Ndx object.
        :param value: a value for the missing scores.

        :return: a Scores object (with the missing scores added and set
                    to value).
        """
        if isinstance(ndx, Key):
            ndx = ndx.to_ndx()

        new_scr = self.align_with_ndx(ndx)
        missing = ndx.trialmask & -new_scr.scoremask
        new_scr.scoremat[missing] = value
        new_scr.scoremask[missing] = True
        assert new_scr.validate(), "Wrong format of Scores"
        return new_scr

    def filter(self, modlist, seglist, keep):
        """Removes some of the information in a Scores object.  Useful for
        creating a gender specific score set from a pooled gender score
        set.  Depending on the value of \'keep\', the two input lists
        indicate the models and test segments (and their associated
        scores) to retain or discard.

        :param modlist: a list of strings which will be compared with
                the modelset of the current Scores object.
        :param seglist: a list of strings which will be compared with
                    the segset of \'inscr\'.
        :param  keep: a boolean indicating whether modlist and seglist are the
                models to keep or discard.

        :return: a filtered version of \'inscr\'.
        """
        if keep:
            keepmods = modlist
            keepsegs = seglist
        else:
            keepmods = diff(self.modelset, modlist)
            keepsegs = diff(self.segset, seglist)

        keepmodidx = np.array(ismember(self.modelset, keepmods))
        keepsegidx = np.array(ismember(self.segset, keepsegs))

        outscr = Scores()
        outscr.modelset = self.modelset[keepmodidx]
        outscr.segset = self.segset[keepsegidx]
        tmp = self.scoremat[np.array(keepmodidx), :]
        outscr.scoremat = tmp[:, np.array(keepsegidx)]
        tmp = self.scoremask[np.array(keepmodidx), :]
        outscr.scoremask = tmp[:, np.array(keepsegidx)]

        assert isinstance(outscr, Scores), 'Wrong Scores format'

        if self.modelset.shape[0] > outscr.modelset.shape[0]:
            logging.info('Number of models reduced from %d to %d', self.modelset.shape[0],
                    outscr.modelset.shape[0])
        if self.segset.shape[0] > outscr.segset.shape[0]:
            logging.info('Number of test segments reduced from %d to %d',
                    self.segset.shape[0], outscr.segset.shape[0])
        return outscr

    def validate(self):
        """Checks that an object of type Scores obeys certain rules that
        must always be true.

            :return: a boolean value indicating whether the object is valid.
        """
        ok = self.scoremat.shape == self.scoremask.shape
        ok &= (self.scoremat.shape[0] == self.modelset.shape[0])
        ok &= (self.scoremat.shape[1] == self.segset.shape[0])
        return ok

    def read(self, inputFileName):
        """Read information from a file and constructs a Scores object. The
	    type of file is deduced from the extension. The extension must be
	    '.txt' for a text file and '.hdf5' or '.h5' for a HDF5 file.

	    :param inputFileName: name of the file o read from
	    """
        extension = os.path.splitext(inputFileName)[1][1:].lower()
        if extension == 'p':
            self.read_pickle(inputFileName)
        elif extension in ['hdf5', 'h5']:
            if h5py_loaded:
                read_scores_hdf5(inputFileName)
            else:
                raise Exception('H5PY is not installed, chose another' +
                        ' format to load your Scores')
        elif extension == 'txt':
            self.read_txt(inputFileName)
        else:
            raise Exception('Error: unknown extension')
        self.sort()

    def read_hdf5(self, inputFileName):
        """Read a Scores object from information in a hdf5 file.

	    :param inputFileName: name of the file to read from
        """
        with h5py.File(inputFileName, "r") as f:

            self.modelset = np.empty(f["/ID/row_ids"].shape, dtype=f["/ID/row_ids"].dtype)
            f["modelset"].read_direct(self.modelset)
            self.modelset = self.modelset.astype('U100', copy=False)

            self.segset = np.empty(f["/ID/column_ids"].shape, dtype=f["/ID/column_ids"].dtype)
            f["segset"].read_direct(self.segset)
            self.segset = self.segset.astype('U100', copy=False)

            self.scoremask = np.empty(f["score_mask"].shape, dtype=f["score_mask"].dtype)
            f["score_mask"].read_direct(self.scoremask)
            self.scoremask = self.scoremask.astype('bool', copy=False)

            self.scoremat = np.empty(f["scores"].shape, dtype=f["scores"].dtype)
            f["scores"].read_direct(self.scoremat)

            assert self.validate(), "Error: wrong Scores format"

    def read_pickle(self, inputFileName):
        """Read Scores in PICKLE format.
        
        :param inputFileName: name of the file to read from
        """
        with gzip.open(inputFileName, "rb" ) as f:
            scores = pickle.load(f)
            self.modelset = scores.modelset
            self.segset = scores.segset
            self.scoremask = scores.scoremask
            self.scoremat = scores.scoremat
            
    def read_txt(self, inputFileName):
        """Creates a Scores object from information stored in a text file.

        :param inputFileName: name of the file to read from
        """
        with open(inputFileName, 'r') as fid:
            lines = [l.rstrip().split() for l in fid]

        models = np.array([], '|O')
        models.resize(len(lines))
        testsegs = np.array([], '|O')
        testsegs.resize(len(lines))
        scores = np.array([])
        scores.resize(len(lines))

        for ii in range(len(lines)):
            models[ii] = lines[ii][0]
            testsegs[ii] = lines[ii][1]
            scores[ii] = float(lines[ii][2])

        modelset = np.unique(models)
        segset = np.unique(testsegs)

        scoremask = np.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
        scoremat = np.zeros((modelset.shape[0], segset.shape[0]))
        for m in range(modelset.shape[0]):
            segs = testsegs[np.array(ismember(models, modelset[m]))]
            scrs = scores[np.array(ismember(models, modelset[m]))]
            idx = segs.argsort()
            segs = segs[idx]
            scrs = scrs[idx]
            scoremask[m, ] = ismember(segset, segs)
            scoremat[m, np.array(ismember(segset, segs))] = scrs

        self.modelset = modelset
        self.segset = segset
        self.scoremask = scoremask
        self.scoremat = scoremat
        assert self.validate(), "Wrong Scores format"
        self.sort()

    def merge(self, scoreList):
        """Merges a list of Scores objects into the current one.
        The resulting must have all models and segment in the input
        Scores (only once) and the union of all the scoremasks.
        It is an error if two of the input Scores objects have a
        score for the same trial.

        :param scoreList: the list of Scores object to merge
        """
        assert isinstance(scoreList, list), "Input is not a list"
        for scr in scoreList:
            assert isinstance(scoreList, list), \
                '{} {} {}'.format("Element ", scr, " is not a Score")

        self.validate()
        for scr2 in scoreList:
            scr_new = Scores()
            scr1 = self
            scr1.sort()
            scr2.sort()

            # create new scr with empty matrices
            scr_new.modelset = np.union1d(scr1.modelset, scr2.modelset)
            scr_new.segset = np.union1d(scr1.segset, scr2.segset)

            # expand scr1 matrices
            scoremat_1 = np.zeros((scr_new.modelset.shape[0],
                                    scr_new.segset.shape[0]))
            scoremask_1 = np.zeros((scr_new.modelset.shape[0],
                                    scr_new.segset.shape[0]), dtype='bool')
            model_index_a = np.argwhere(np.in1d(scr_new.modelset,
                                                scr1.modelset))
            model_index_b = np.argwhere(np.in1d(scr1.modelset,
                                                scr_new.modelset))
            seg_index_a = np.argwhere(np.in1d(scr_new.segset,
                                                scr1.segset))
            seg_index_b = np.argwhere(np.in1d(scr1.segset, scr_new.segset))
            scoremat_1[model_index_a[:, None], seg_index_a] \
                    = scr1.scoremat[model_index_b[:, None], seg_index_b]
            scoremask_1[model_index_a[:, None], seg_index_a] \
                    = scr1.scoremask[model_index_b[:, None], seg_index_b]

            # expand scr2 matrices
            scoremat_2 = np.zeros((scr_new.modelset.shape[0],
                                    scr_new.segset.shape[0]))
            scoremask_2 = np.zeros((scr_new.modelset.shape[0],
                                    scr_new.segset.shape[0]), dtype='bool')
            model_index_a = np.argwhere(np.in1d(scr_new.modelset,
                                                scr2.modelset))
            model_index_b = np.argwhere(np.in1d(scr2.modelset,
                                                scr_new.modelset))
            seg_index_a = np.argwhere(np.in1d(scr_new.segset, scr2.segset))
            seg_index_b = np.argwhere(np.in1d(scr2.segset, scr_new.segset))
            scoremat_2[model_index_a[:, None], seg_index_a] \
                    = scr2.scoremat[model_index_b[:, None], seg_index_b]
            scoremask_2[model_index_a[:, None], seg_index_a] \
                    = scr2.scoremask[model_index_b[:, None], seg_index_b]

            # check for clashes
            assert np.sum(scoremask_1 & scoremask_2) == 0, \
                    "Conflict in the new scoremask"

            # merge masks
            self.scoremat = scoremat_1 + scoremat_2
            self.scoremask = scoremask_1 | scoremask_2
            self.modelset = scr_new.modelset
            self.segset = scr_new.segset
            assert self.validate(), 'Wrong Scores format'

    def sort(self):
        """Sort models and segments"""
        sort_model_idx = np.argsort(self.modelset)
        sort_seg_idx = np.argsort(self.segset)
        sort_mask = self.scoremask[sort_model_idx[:, None], sort_seg_idx]
        sort_mat = self.scoremat[sort_model_idx[:, None], sort_seg_idx]
        self.modelset.sort()
        self.segset.sort()
        self.scoremat = sort_mat
        self.scoremask = sort_mask
    
    def get_score(self, modelID, segID):
        """return a score given a model and segment identifiers
        raise an error if the trial does not exist
        :param modelID: id of the model
        :param segID: id of the test segment
        """
        model_idx = np.argwhere(self.modelset == modelID)
        seg_idx = np.argwhere(self.segset == segID)
        if model_idx.shape[0] == 0:
            raise Exception('No such model as: %s', modelID)
        elif seg_idx.shape[0] == 0:
            raise Exception('No such segment as: %s', segID)
        else:
            return self.scoremat[model_idx, seg_idx]



