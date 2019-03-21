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
Copyright 2014-2019 Anthony Larcher


The authors would like to thank the BUT Speech@FIT group (http://speech.fit.vutbr.cz) and Lukas BURGET
for sharing the source code that strongly inspired this module. Thank you for your valuable contribution.
"""

import h5py
import numpy
import torch

from sidekit.frontend.io import _read_dataset_percentile
from sidekit.features_server import FeaturesServer


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


from collections import OrderedDict
from torch.utils.data import Dataset


def read_batch(batch_file):
    with h5py.File(batch_file, 'r') as h5f:
        data = _read_dataset_percentile(h5f, 'data')
        label = h5f['label'].value

        # Normalize and reshape
        data = data.reshape((len(label), data.shape[0] // len(label), data.shape[1])).transpose(0, 2, 1)
        for idx in range(data.shape[0]):
            m = data[idx].mean(axis=0)
            s = data[idx].std(axis=0)
            data[idx] = (data[idx] - m) / s
        return data, label

def read_hot_batch(batch_file, spk_nb):
    with h5py.File(batch_file, 'r') as h5f:
        data = _read_dataset_percentile(h5f, 'data')
        label = h5f['label'].value

        # Normalize and reshape
        data = data.reshape((len(label), data.shape[0] // len(label), data.shape[1])).transpose(0, 2, 1)
        for idx in range(data.shape[0]):
            m = data[idx].mean(axis=0)
            s = data[idx].std(axis=0)
            data[idx] = (data[idx] - m) / s

        lbl = numpy.zeros((128, spk_nb))
        lbl[numpy.arange(128), label] += 1
        return data, lbl

class XvectorDataset(Dataset):
    """
    Object that takes a list of files from a file and initialize a Dataset
    """
    def __init__(self, batch_list, batch_path):
        with open(batch_list, 'r') as fh:
            self.batch_files = [batch_path + '/' + l.rstrip() for l in fh]
        self.len = len(self.batch_files)

    def __getitem__(self, index):
        data, label = read_batch(self.batch_files[index])
        return torch.from_numpy(data).type(torch.FloatTensor), torch.from_numpy(label.astype('long'))

    def __len__(self):
        return self.len

class XvectorHotDataset(Dataset):
    """
    Object that takes a list of files from a file and initialize a Dataset
    """
    def __init__(self, batch_list, batch_path, spk_nb):
        with open(batch_list, 'r') as fh:
            self.batch_files = [batch_path + '/' + l.rstrip() for l in fh]
        self.len = len(self.batch_files)
        self.spk_nb = spk_nb

    def __getitem__(self, index):
        data, label = read_hot_batch(self.batch_files[index], self.spk_nb)
        return torch.from_numpy(data).type(torch.FloatTensor), torch.from_numpy(label.astype(numpy.float32))

    def __len__(self):
        return self.len

class XvectorMultiDataset(Dataset):
    """
    Object that takes a list of files as a Python List and initialize a DataSet
    """
    def __init__(self, batch_list, batch_path):
        self.batch_files = [batch_path + '/' + l for l in batch_list]
        self.len = len(self.batch_files)

    def __getitem__(self, index):
        data, label = read_batch(self.batch_files[index])
        return torch.from_numpy(data).type(torch.FloatTensor), torch.from_numpy(label.astype('long'))

    def __len__(self):
        return self.len

class XvectorMultiDataset_hot(Dataset):
    """
    Object that takes a list of files as a Python List and initialize a DataSet
    """
    def __init__(self, batch_list, batch_path, spk_nb):
        self.batch_files = [batch_path + '/' + l for l in batch_list]
        self.len = len(self.batch_files)
        self.spk_nb = spk_nb

    def __getitem__(self, index):
        data, label = read_hot_batch(self.batch_files[index], self.spk_nb)
        return torch.from_numpy(data).type(torch.FloatTensor), torch.from_numpy(label.astype(numpy.float32))

    def __len__(self):
        return self.len

class StatDataset(Dataset):
    """
    Object that initialize a Dataset from an sidekit.IdMap
    """
    def __init__(self, idmap, fs_param):
        self.idmap = idmap
        self.fs = FeaturesServer(**fs_param)
        self.len = self.idmap.leftids.shape[0]

    def __getitem__(self, index):
        data, _ = self.fs.load(self.idmap.rightids[index])
        data = (data - data.mean(0)) / data.std(0)
        data = data.reshape((1, data.shape[0], data.shape[1])).transpose(0, 2, 1).astype(numpy.float32)
        return self.idmap.leftids[index], self.idmap.rightids[index], torch.from_numpy(data).type(torch.FloatTensor)


    def __len__(self):
        return self.len


