# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:30:05 2014

@author: antho
"""

__author__ = "Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brummer", "Edward de Villiers"]

import numpy as np
import os
import sys
import pickle
import gzip
import logging
try: 
    import h5py 
    h5py_loaded = True
except ImportError: 
    h5py_loaded = False 

def diff(list1, list2):
    c = [item for item in list1 if item not in list2]
    c.sort()
    return c


def ismember(list1, list2):
    c = [item in list2 for item in list1]
    return c

#if h5py_loaded:
#    
#    def save_ndx_hdf5(ndx, outpuFileName):
#        """ Save Ndx object in HDF5 format
#
#       :param ndx: Ndx object to save 
#	 :param outputFileName: name of the file to write to
#        """
#        set_model = "/ID/row_ids"
#        set_seg = "/ID/column_ids"
#        set_mask = "/trial_mask"
#        if sys.hexversion >= 0x03000000:
#            outpuFileName = outpuFileName.encode()
#            set_model = set_model.encode()
#            set_seg = set_seg.encode()
#            set_mask = set_mask.encode()
#
#        fid = h5py.h5f.create(outpuFileName)
#        filetype = h5py.h5t.FORTRAN_S1.copy()
#        filetype.set_size(h5py.h5t.VARIABLE)
#        memtype = h5py.h5t.C_S1.copy()
#        memtype.set_size(h5py.h5t.VARIABLE)
#
#        h5py.h5g.create(fid, '/ID')
#
#        space_model = h5py.h5s.create_simple(ndx.modelset.shape)
#        dset_model = h5py.h5d.create(fid, set_model, filetype, space_model)
#        dset_model.write(h5py.h5s.ALL, h5py.h5s.ALL, ndx.modelset)
#
#        space_seg = h5py.h5s.create_simple(ndx.segset.shape)
#        dset_seg = h5py.h5d.create(fid, set_seg, filetype, space_seg)
#        dset_seg.write(h5py.h5s.ALL, h5py.h5s.ALL, ndx.segset)
#
#        space_mask = h5py.h5s.create_simple(ndx.trialmask.shape)
#        dset_mask = h5py.h5d.create(fid, set_mask,
#                                    h5py.h5t.NATIVE_INT8, space_mask)
#        dset_mask.write(h5py.h5s.ALL, h5py.h5s.ALL,
#                                    np.ascontiguousarray(ndx.trialmask))
#
#        # Close and release resources.
#        fid.close()
#        del space_model
#        del dset_model
#        del space_mask
#        del dset_mask
#        del space_seg
#        del dset_seg
#        del fid
#
#    def read_ndx_hdf5(ndx, inputFileName):
#        """Creates an Ndx object from the information in an hdf5 file.
#
#        :param ndx: Ndx object to load
#        :param inputFileName: name of the file to read from
#        """
#        fid = h5py.h5f.open(inputFileName)
#
#        set_model = h5py.h5d.open(fid, "/ID/row_ids")
#        ndx.modelset = np.empty(set_model.shape[0], dtype=set_model.dtype)
#        set_model.read(h5py.h5s.ALL, h5py.h5s.ALL, ndx.modelset)
#
#        set_seg = h5py.h5d.open(fid, "/ID/column_ids")
#        ndx.segset = np.empty(set_seg.shape[0], dtype=set_seg.dtype)
#        set_seg.read(h5py.h5s.ALL, h5py.h5s.ALL, ndx.segset)
#
#        set_mask = h5py.h5d.open(fid, "/trial_mask")
#        ndx.trialmask.resize(set_mask.shape)
#        rdata = np.zeros(set_mask.shape, dtype=np.int8)
#        set_mask.read(h5py.h5s.ALL, h5py.h5s.ALL, rdata)
#        ndx.trialmask = rdata.astype('bool')
#
#        rdata = np.zeros(set_mask.shape, dtype=np.int)
#        fid.close()

class Ndx:
    """A class that encodes trial index information.  It has a list of
    model names and a list of test segment names and a matrix
    indicating which combinations of model and test segment are
    trials of interest.
    
    :attr modelset: list of unique models in a ndarray
    :attr segset:  list of unique test segments in a ndarray
    :attr trialmask: 2D ndarray of boolean. Rows correspond to the models 
            and columns to the test segments. True if the trial is of interest.
    """

    def __init__(self, ndxFileName='', ndxFileFormat='hdf5',
                 models=np.array([]), testsegs=np.array([])):
        """Initialize a Ndx object by loading information from a file
        in PICKLE, HDF5 or text.

        :param ndxFileName: name of the file to load
        :param ndxFileFormat: format of the file to load. Can be:
	        
        - 'pickle'
        - 'hdf5'
        - 'txt

	    Default is 'hdf5'.
        """
        self.modelset = np.empty(0, dtype="|O")
        self.segset = np.empty(0, dtype="|O")
        self.trialmask = np.array([], dtype="bool")

        if not h5py_loaded:
            ndxFileFormat = 'pickle'

        if ndxFileName == '':
            modelset = np.unique(models)
            segset = np.unique(testsegs)
    
            trialmask = np.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
            for m in range(modelset.shape[0]):
                segs = testsegs[np.array(ismember(models, modelset[m]))]
                trialmask[m, ] = ismember(segset, segs)
    
            self.modelset = modelset
            self.segset = segset
            self.trialmask = trialmask
            assert self.validate(), "Wrong Ndx format"
            
        elif ndxFileFormat == 'pickle':
            self.read_pickle(ndxFileName)
        elif ndxFileFormat == 'hdf5':
            if h5py_loaded:
                self.read_hdf5(ndxFileName)
            else:
                raise Exception('H5PY is not installed, chose another' +
                        ' format to load your Ndx')
        elif ndxFileFormat == 'txt':
            self.read_txt(ndxFileName)
        else:
            raise Exception('Wrong ndxFileFormat')

    def save(self, outputFileName):
        """Save the Ndx object to file. The format of the file is deduced from
        the extension of the filename. The format can be PICKLE, HDF5 or text.
        Extension for pickle should be '.p', text file should be '.txt' 
        and for HDF5 it should be 	'.hdf5' or '.h5'

        :param outputFileName: name of the file to write to	
        """
        extension = os.path.splitext(outputFileName)[1][1:].lower()
        if extension == 'p':
            self.save_pickle(outputFileName)
        elif extension.lower() in ['hdf5', 'h5']:
            if h5py_loaded:
                self.save_hdf5(outputFileName)
            else:
                raise Exception('H5PY is not installed, chose another' +
                        ' format to save your Ndx')
        elif extension == 'txt':
            self.save_txt(outputFileName)
        else:
            raise Exception('Error: unknown extension')

    def save_hdf5(self, outpuFileName):
        """ Save Ndx object in HDF5 format

	 :param outputFileName: name of the file to write to
        """
        set_model = "/ID/row_ids"
        set_seg = "/ID/column_ids"
        set_mask = "/trial_mask"
        if sys.hexversion >= 0x03000000:
            outpuFileName = outpuFileName.encode()
            set_model = set_model.encode()
            set_seg = set_seg.encode()
            set_mask = set_mask.encode()

        fid = h5py.h5f.create(outpuFileName)
        filetype = h5py.h5t.FORTRAN_S1.copy()
        filetype.set_size(h5py.h5t.VARIABLE)
        memtype = h5py.h5t.C_S1.copy()
        memtype.set_size(h5py.h5t.VARIABLE)

        h5py.h5g.create(fid, '/ID')

        space_model = h5py.h5s.create_simple(self.modelset.shape)
        dset_model = h5py.h5d.create(fid, set_model, filetype, space_model)
        dset_model.write(h5py.h5s.ALL, h5py.h5s.ALL, self.modelset)

        space_seg = h5py.h5s.create_simple(self.segset.shape)
        dset_seg = h5py.h5d.create(fid, set_seg, filetype, space_seg)
        dset_seg.write(h5py.h5s.ALL, h5py.h5s.ALL, self.segset)

        space_mask = h5py.h5s.create_simple(self.trialmask.shape)
        dset_mask = h5py.h5d.create(fid, set_mask,
                                    h5py.h5t.NATIVE_INT8, space_mask)
        dset_mask.write(h5py.h5s.ALL, h5py.h5s.ALL,
                                    np.ascontiguousarray(self.trialmask))

        # Close and release resources.
        fid.close()
        del space_model
        del dset_model
        del space_mask
        del dset_mask
        del space_seg
        del dset_seg
        del fid

    def save_pickle(self, outputFileName):
        """Save Ndx in PICKLE format
        
        :param outputFilename: name of the file to write to
        """
        with gzip.open(outputFileName, "wb" ) as f:
            pickle.dump( self, f)

    def save_txt(self, outputFileName):
        """Save a Ndx object in a text file
	
	:param outputFileName: name of the file to write to
	"""
        fid = open(outputFileName, 'w')
        for m in range(self.modelset.shape[0]):
            segs = self.segset[self.trialmask[m, ]]
            for s in segs:
                fid.write('{} {}\n'.format(self.modelset[m], s))
        fid.close()

    def filter(self, modlist, seglist, keep):
        """Removes some of the information in an Ndx. Useful for creating a
	gender specific Ndx from a pooled gender Ndx.  Depending on the
	value of \'keep\', the two input lists indicate the strings to
	retain or the strings to discard.
	
	:param modlist: a cell array of strings which will be compared with
	        the modelset of 'inndx'.
	:param seglist: a cell array of strings which will be compared with
	        the segset of 'inndx'.
	:param keep: a boolean indicating whether modlist and seglist are the
	        models to keep or discard. 

	:return: a filtered version of the current Ndx object.
	"""
        if keep:
            keepmods = modlist
            keepsegs = seglist
        else:
            keepmods = diff(self.modelset, modlist)
            keepsegs = diff(self.segset, seglist)

        outndx = Ndx()

        keepmodidx = np.array(ismember(self.modelset, keepmods))
        keepsegidx = np.array(ismember(self.segset, keepsegs))

        outndx = Ndx()
        outndx.modelset = self.modelset[keepmodidx]
        outndx.segset = self.segset[keepsegidx]
        tmp = self.trialmask[np.array(keepmodidx), :]
        outndx.trialmask = tmp[:, np.array(keepsegidx)]

        assert outndx.validate, "Wrong Ndx format"

        if self.modelset.shape[0] > outndx.modelset.shape[0]:
            logging.info('Number of models reduced from %d to %d', self.modelset.shape[0],
                        outndx.modelset.shape[0])
        if self.segset.shape[0] > outndx.segset.shape[0]:
            logging.info('Number of test segments reduced from %d to %d',
                        self.segset.shape[0], outndx.segset.shape[0])
        return outndx

    def validate(self):
        """Checks that an object of type Ndx obeys certain rules that
	% must always be true.
	
	:return: a boolean value indicating whether the object is valid
	"""
        ok = isinstance(self.modelset, np.ndarray)
        ok = ok & isinstance(self.segset, np.ndarray)
        ok = ok & isinstance(self.trialmask, np.ndarray)

        ok = ok & (self.modelset.ndim == 1)
        ok = ok & (self.segset.ndim == 1)
        ok = ok & (self.trialmask.ndim == 2)

        ok = ok & (self. trialmask.shape ==
                    (self.modelset.shape[0], self.segset.shape[0]))
        return ok

    def read(self, inputFileName):
        """Reads information from a file and constructs an Ndx object.  The
	type of file is deduced from the extension. The extension must be 
	'.txt' for a text file and '.hdf5' or '.h5' for a HDF5 file.

	:param inputFileName: name of the file to read from
	"""
        extension = os.path.splitext(inputFileName)[1][1:].lower()
        if extension == 'p':
            self.read_pickle(inputFileName)
        elif extension in ['hdf5', 'h5']:
            if h5py_loaded:
                read_ndx_hdf5(inputFileName)
            else:
                raise Exception('H5PY is not installed, chose another' +
                        ' format to load your Ndx')
        elif extension == 'txt':
            self.read_txt(inputFileName)
        else:
            raise Exception('Error: unknown extension')

    def read_hdf5(self, inputFileName):
        """Creates an Ndx object from the information in an hdf5 file.

        :param inputFileName: name of the file to read from
        """
        fid = h5py.h5f.open(inputFileName)

        set_model = h5py.h5d.open(fid, "/ID/row_ids")
        self.modelset = np.empty(set_model.shape[0], dtype=set_model.dtype)
        set_model.read(h5py.h5s.ALL, h5py.h5s.ALL, self.modelset)

        set_seg = h5py.h5d.open(fid, "/ID/column_ids")
        self.segset = np.empty(set_seg.shape[0], dtype=set_seg.dtype)
        set_seg.read(h5py.h5s.ALL, h5py.h5s.ALL, self.segset)

        set_mask = h5py.h5d.open(fid, "/trial_mask")
        self.trialmask.resize(set_mask.shape)
        rdata = np.zeros(set_mask.shape, dtype=np.int8)
        set_mask.read(h5py.h5s.ALL, h5py.h5s.ALL, rdata)
        self.trialmask = rdata.astype('bool')

        rdata = np.zeros(set_mask.shape, dtype=np.int)
        fid.close()

    def read_pickle(self, inputFileName):
        """Read Ndx in PICKLE format.
        
        :param inputFileName: name of the file to read from
        """
        with gzip.open(inputFileName, "rb" ) as f:
            ndx = pickle.load(f)
            self.modelset = ndx.modelset
            self.segset = ndx.segset
            self.trialmask = ndx.trialmask

    def read_txt(self, inputFileName):
        """Creates an Ndx object from information stored in a text file.

	:param inputFileName: name of the file to read from
	"""
        with open(inputFileName, 'r') as fid:
            lines = [l.rstrip().split() for l in fid]

        models = np.array([], '|O')
        models.resize(len(lines))
        testsegs = np.array([], '|O')
        testsegs.resize(len(lines))
        for ii in range(len(lines)):
            models[ii] = lines[ii][0]
            testsegs[ii] = lines[ii][1]

        modelset = np.unique(models)
        segset = np.unique(testsegs)

        trialmask = np.zeros((modelset.shape[0], segset.shape[0]), dtype="bool")
        for m in range(modelset.shape[0]):
            segs = testsegs[np.array(ismember(models, modelset[m]))]
            trialmask[m, ] = ismember(segset, segs)

        self.modelset = modelset
        self.segset = segset
        self.trialmask = trialmask
        assert self.validate(), "Wrong Ndx format"

    def merge(self, ndxList):
        """Merges a list of Ndx objects into the current one.
        The resulting ndx must have all models and segment in the input
        ndxs (only once).  A trial in any ndx becomes a trial in the
        output ndx

        :param ndxList: list of Ndx objects to merge
	  """
        assert isinstance(ndxList, list), "Input is not a list"
        for ndx in ndxList:
            assert isinstance(ndxList, list), \
                '{} {} {}'.format("Element ", ndx, " is not an Ndx")

        self.validate()
        for ndx2 in ndxList:
            ndx_new = Ndx()
            ndx1 = self

            # create new ndx with empty masks
            ndx_new.modelset = np.union1d(ndx1.modelset, ndx2.modelset)
            ndx_new.segset = np.union1d(ndx1.segset, ndx2.segset)

            # expand ndx1 mask
            trials_1 = np.zeros((ndx_new.modelset.shape[0],
                                ndx_new.segset.shape[0]),
                                dtype="bool")
            model_index_a = np.argwhere(np.in1d(ndx_new.modelset,
                                                ndx1.modelset))
            model_index_b = np.argwhere(np.in1d(ndx1.modelset,
                                                ndx_new.modelset))
            seg_index_a = np.argwhere(np.in1d(ndx_new.segset, ndx1.segset))
            seg_index_b = np.argwhere(np.in1d(ndx1.segset, ndx_new.segset))
            trials_1[model_index_a[:, None], seg_index_a] \
                        = ndx1.trialmask[model_index_b[:, None], seg_index_b]

            # expand ndx2 mask
            trials_2 = np.zeros((ndx_new.modelset.shape[0],
                                ndx_new.segset.shape[0]),
                                dtype="bool")
            model_index_a = np.argwhere(np.in1d(ndx_new.modelset,
                                                ndx2.modelset))
            model_index_b = np.argwhere(np.in1d(ndx2.modelset,
                                                ndx_new.modelset))
            seg_index_a = np.argwhere(np.in1d(ndx_new.segset, ndx2.segset))
            seg_index_b = np.argwhere(np.in1d(ndx2.segset, ndx_new.segset))
            trials_2[model_index_a[:, None], seg_index_a] \
                        = ndx2.trialmask[model_index_b[:, None], seg_index_b]

            # merge masks
            trials = trials_1 | trials_2

            # build new ndx
            ndx_new.trialmask = trials
            self.modelset = ndx_new.modelset
            self.segset = ndx_new.segset
            self.trialmask = ndx_new.trialmask

    def clean(self, enroll, featureDir, featureExtension):
        """Clean the Ndx by removing missing models and segments
	
        :param enroll: an IdMap with the defition of each model from the Ndx
        :param featureDir: directory where the feature files are to be find
        :param featureExtension: extension of the feature files to look for
        """
        #TODO
        pass






