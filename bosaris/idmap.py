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
This is the 'idmap' module
"""
import os.path
import sys
import numpy as np
import pickle
import gzip
import logging
import copy
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


class IdMap:
    """A class that stores a map between identifiers (strings).  One
    list is called 'leftids' and the other 'rightids'.  The class
    provides methods that convert a sequence of left ids to a
    sequence of right ids and vice versa.  If 'leftids' or 'rightids'
    contains duplicates then all occurrences are used as the index
    when mapping.

    :attr leftids: a list of classes in a ndarray
    :attr rightids: a list of segments in a ndarray
    :attr start: index of the first frame of the segment
    :attr stop: index of the last frame of the segment
    """

    def __init__(self, idmapFileName='', idmapFileFormat='hdf5'):
        """Initialize an IdMap object

        :param idmapFileName: name of a file to load. Default is ''.
        :param idmapFileFormat: format of the file to load. Can be:
            - 'pickle'
            - 'hdf5' (default)
            - 'txt'
        In case the idmapFileName is empty, initialize an empty IdMap object.
        """
        self.leftids = np.empty(0, dtype="|O")
        self.rightids = np.empty(0, dtype="|O")
        self.start = np.empty(0, dtype="|O")
        self.stop = np.empty(0, dtype="|O")

        if idmapFileName == '':
            pass
        elif idmapFileFormat.lower() == 'pickle':
            self.read_pickle(idmapFileName)
        elif idmapFileFormat.lower() in ['hdf5', 'h5']:
            if h5py_loaded:
                self.read_hdf5(idmapFileName)
            else:
                raise Exception('h5py is not installed, chose another' + ' format to load your IdMap')
        elif idmapFileFormat.lower() == 'txt':
            self.read_txt(idmapFileName)
        else:
            raise Exception('Wrong output format, must be pickle, hdf5 or txt')

    @check_path_existance
    def save(self, outputFileName):
        """Save the IdMap object to file. The format of the file 
        to create is set accordingly to the extension of the filename.
        This extension can be '.p' for pickle format, '.txt' for text format 
        and '.hdf5' or '.h5' for HDF5 format.

        :param outputFileName: name of the file to write to
        
        :warning: hdf5 format save only leftids and rightids
        """
        extension = os.path.splitext(outputFileName)[1][1:].lower()
        if extension == 'p':
            self.save_pickle(outputFileName)
        elif extension in ['hdf5', 'h5']:
            if h5py_loaded:
                self.save_hdf5(outputFileName)
            else:
                raise Exception('h5py is not installed, chose another' + 
                        ' format to load your IdMap')
        elif extension == 'txt':
            self.save_txt(outputFileName)
        else:
            raise Exception('Wrong output format, must be pickle, hdf5 or txt')

    @check_path_existance
    def save_hdf5(self, outpuFileName):
        """ Save IdMap in HDF5 format

        :param outpuFileName: name of the file to write to
        """
        assert self.validate(), "Error: wrong IdMap format"
        with h5py.File(outpuFileName, "w") as f:
            f.create_dataset("leftids", data=self.leftids.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("rightids", data=self.rightids.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            # WRITE START and STOP
            start = copy.deepcopy(self.start)
            start[np.isnan(self.start.astype('float'))] = -1
            start = start.astype('int8', copy=False)

            stop = copy.deepcopy(self.stop)
            stop[np.isnan(self.stop.astype('float'))] = -1
            stop = stop.astype('int8', copy=False)

            f.create_dataset("start", data=start,
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("stop", data=stop,
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)

    @check_path_existance
    def save_pickle(self, outputFileName):
        """Save IdMap in PICKLE format
        
        :param outputFileName: name of the file to write to
        """
        with gzip.open(outputFileName, "wb" ) as f:
            pickle.dump( self, f)

    @check_path_existance
    def save_txt(self, outputFileName):
        """Saves the Id_Map to a text file.
        
        :param outputFileName: name of the output text file
        """
        with open(outputFileName, 'w') as outputFile:
            for left, right, start, stop in zip(self.leftids, self.rightids,
                                            self.start, self.stop):
                line = ' '.join(filter(None, (left, right, start, stop))) + '\n'
                outputFile.write(line)

    def map_left_to_right(self, leftidlist):
        """Maps an array of ids to a new array of ids using the given map.  
        The input ids are matched against the leftids of the map and the
        output ids are taken from the corresponding rightids of the map.
        
        Beware: if leftids are not unique in the IdMap, only the last value 
        corresponding is kept

        :param leftidlist: an array of strings to be matched against the
            leftids of the idmap.  The rightids corresponding to these
            leftids will be returned.

        :return: an array of strings that are the mappings of the
            strings in leftidlist.
        """
        tmpDict = dict(zip(self.leftids, self.rightids))
        inter = np.intersect1d(self.leftids, leftidlist)
        rightids = np.empty(inter.shape[0], '|O')
        
        idx = 0
        for left in leftidlist:
            if left in inter:
                rightids[idx] = tmpDict[left]
                idx += 1

        lostIds = np.unique(leftidlist).shape[0] - inter.shape[0]
        if lostIds:
            logging.warning('{} ids could not be mapped'.format(lostIds))

        return rightids

    def map_right_to_left(self, rightidlist):
        """Maps an array of ids to a new array of ids using the given map.  
        The input ids are matched against the rightids of the map and the
        output ids are taken from the corresponding leftids of the map.

        Beware: if rightids are not unique in the IdMap, only the last value 
        corresponding is kept

        :param rightidlist: An array of strings to be matched against the
            rightids of the idmap.  The leftids corresponding to these
            rightids will be returned.

        :return: an array of strings that are the mappings of the
            strings in rightidlist.
        """
        tmpDict = dict(zip(self.rightids, self.leftids))
        inter = np.intersect1d(self.rightids, rightidlist)
        leftids = np.empty(inter.shape[0], '|O')
        
        idx = 0
        for right in rightidlist:
            if right in inter:
                leftids[idx] = tmpDict[right]
                idx += 1        
        
        lostIds = np.unique(rightidlist).shape[0] - inter.shape[0]
        if lostIds:
            logging.warning('{} ids could not be mapped'.format(lostIds))

        return leftids

    def filter_on_left(self, idlist, keep):
        """Removes some of the information in an idmap.  Depending on the
        value of 'keep', the idlist indicates the strings to retain or
        the strings to discard.

        :param idlist: an array of strings which will be compared with
            the leftids of the current.
        :param keep: A boolean indicating whether idlist contains the ids to
            keep or to discard.

        :return: a filtered version of the current IdMap.
        """
        # get the list of ids to keep
        if keep:
            keepids = np.unique(idlist)
        else:
            keepids = np.setdiff1d(self.leftids, idlist)
        
        keep_idx = np.in1d(self.leftids, keepids)
        out_idmap = IdMap()
        out_idmap.leftids = self.leftids[keep_idx]
        out_idmap.rightids = self.rightids[keep_idx]
        out_idmap.start = self.start[keep_idx]
        out_idmap.stop = self.stop[keep_idx]
        
        return out_idmap

    def filter_on_right(self, idlist, keep):
        """Removes some of the information in an idmap.  Depending on the
        value of 'keep', the idlist indicates the strings to retain or
        the strings to discard.

        :param idlist: an array of strings which will be compared with
            the rightids of the current IdMap.
        :param keep: a boolean indicating whether idlist contains the ids to
            keep or to discard.

        :return: a filtered version of the current IdMap.
        """
        # get the list of ids to keep
        if keep:
            keepids = np.unique(idlist)
        else:
            keepids = np.setdiff1d(self.rightids, idlist)        
        
        keep_idx = np.in1d(self.rightids, keepids)
        out_idmap = IdMap()
        out_idmap.leftids = self.leftids[keep_idx]
        out_idmap.rightids = self.rightids[keep_idx]
        out_idmap.start = self.start[keep_idx]
        out_idmap.stop = self.stop[keep_idx]
        return out_idmap

    def validate(self, warn=False):
        """Checks that an object of type Id_Map obeys certain rules that
        must alows be true.
        
        :param warn: boolean. If True, print a warning if strings are
            duplicated in either left or right array

        :return: a boolean value indicating whether the object is valid.

        """
        ok = (self.leftids.shape
                == self.rightids.shape
                == self.start.shape
                == self.stop.shape) \
                & self.leftids.ndim == 1
                
        if warn & (self.leftids.shape != np.unique(self.leftids).shape):
            logging.warning('The left id list contains duplicate identifiers')
        if warn & (self.rightids.shape != np.unique(self.rightids).shape):
            logging.warning('The right id list contains duplicate identifiers')
        return ok

    def set(self, left, right):
        self.leftids = left
        self.rightids = right
        self.start = np.empty(self.rightids.shape, '|O')
        self.stop = np.empty(self.rightids.shape, '|O')

    def read(self, inputFileName):
        """Read an IdMap object from a file.The format of the file to read from
        is determined by the extension of the filename.
        This extension can be '.p' for pickle format,
        '.txt' for text format and '.hdf5' or '.h5' for HDF5 format.

        :param inputFileName: name of the file to read from
        """
        extension = os.path.splitext(inputFileName)[1][1:].lower()
        if extension == 'p':
            self.read_pickle(inputFileName)
        elif extension in ['hdf5', 'h5']:
            if h5py_loaded:
                self.read_hdf5(inputFileName)
        elif extension == 'txt':
            self.read_txt(inputFileName)
        else:
            raise Exception('Wrong input format, must be pickle, hdf5 or txt')

    def read_hdf5(self, inputFileName):
        """Read IdMap in hdf5 format.

        :param inputFileName: name of the file to read from
        """
        with h5py.File(inputFileName, "r") as f:
            self.leftids = f.get("leftids").value
            self.rightids = f.get("rightids").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                self.leftids = self.leftids.astype('U100', copy=False)
                self.rightids = self.rightids.astype('U100', copy=False)

            tmpstart = f.get("start").value
            tmpstop = f.get("stop").value
            self.start = np.empty(f["start"].shape, '|O')
            self.stop = np.empty(f["stop"].shape, '|O')
            self.start[tmpstart != -1] = tmpstart[tmpstart != -1]
            self.stop[tmpstop != -1] = tmpstop[tmpstop != -1]

            assert self.validate(), "Error: wrong IdMap format"

    def read_pickle(self, inputFileName):
        """Read IdMap in PICKLE format.
        
        :param inputFileName: name of the file to read from
        """
        with gzip.open(inputFileName, "rb") as f:
            idmap = pickle.load(f)
            self.leftids = idmap.leftids
            self.rightids = idmap.rightids
            self.start = idmap.start
            self.stop = idmap.stop

    def read_txt(self, inputFileName):
        """Read IdMap in text format.

        :param inputFileName: name of the file to read from
        """
        with file(inputFileName) as f:
            columns = len(f.readline().split(' '))

        if columns == 2:
            self.leftids, self.rightids = np.loadtxt(inputFileName, 
                    dtype={'names': ('left', 'right'),'formats': ('|O', '|O')}, 
                    usecols=(0, 1), unpack=True)
            self.start = np.empty(self.rightids.shape, '|O')
            self.stop = np.empty(self.rightids.shape, '|O')
        
        # If four columns
        elif columns == 4:
            self.leftids, self.rightids, self.start, self.stop  = np.loadtxt(
                    inputFileName, 
                    dtype={'names': ('left', 'right', 'start', 'stop'),
                    'formats': ('|O', '|O', 'int', 'int')}, unpack=True)
    
        if not self.validate():
            raise Exception('Wrong format of IdMap')

    def merge(self, idmap2):
        """ Merges the current IdMap with another IdMap or a list of IdMap objects..

        :param idmap2: Another Id_Map object.

        :return: an Id_Map object that contains the information from the two
            input Id_Maps.
        """
        idmap = IdMap()
        if self.validate() & idmap2.validate():
            # verify that both IdMap don't share any id
            if (np.intersect1d(self.leftids, idmap2.leftids).size &
                np.intersect1d(self.rightids, idmap2.rightids).size):
            
                idmap.leftids = np.concatenate((self.leftids,
                                            idmap2.leftids), axis=0)
                idmap.rightids = np.concatenate((self.rightids,
                                             idmap2.rightids), axis=0)
                idmap.start = np.concatenate((self.start,
                                             idmap2.start), axis=0) 
                idmap.stop = np.concatenate((self.stop,
                                             idmap2.stop), axis=0)
            else:
                raise Exception('Idmaps being merged share ids.')
        else:
            raise Exception('Cannot merge IdMaps, wrong type')

        if not idmap.validate():
            raise Exception('Wrong format of IdMap')

        return idmap

