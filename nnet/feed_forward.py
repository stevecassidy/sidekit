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
Copyright 2014-2017 Anthony Larcher

:mod:`theano_utils` provides utilities to facilitate the work with SIDEKIT
and THEANO.

The authors would like to thank the BUT Speech@FIT group (http://speech.fit.vutbr.cz) and Lukas BURGET
for sharing the source code that strongly inspired this module. Thank you for your valuable contribution.
"""
import copy
import ctypes
import h5py
import logging
import multiprocessing
import numpy
import os
import time
import torch
import warnings

import sidekit.frontend
from sidekit.sidekit_io import init_logging
from sidekit.sidekit_wrappers import check_path_existance

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2017 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def kaldi_to_hdf5(input_file_name, output_file_name):
    """
    Convert a text file containing frame alignment from Kaldi into an
    HDF5 file with the following structure:

        show/start/labels

    :param input_file_name:
    :param output_file_name:
    :return:
    """
    with open(input_file_name, "r") as fh:
        lines = [line.rstrip() for line in fh]

    with h5py.File(output_file_name, "w") as h5f:
        for line in lines[1:-1]:
            show = line.split('_')[0] + '_' + line.split('_')[1]
            start = int(line.split('_')[2].split('-')[0])
            label = numpy.array([int(x) for x in line.split()[1:]], dtype="int16")
            h5f.create_dataset(show + "/{}".format(start), data=label,
                               maxshape=(None,),
                               compression="gzip",
                               fletcher32=True)


def segment_mean_std_hdf5(input_segment):
    """
    Compute the sum and square sum of all features for a list of segments.
    Input files are in HDF5 format

    :param input_segment: list of segments to read from, each element of the list is a tuple of 5 values,
        the filename, the index of thefirst frame, index of the last frame, the number of frames for the
        left context and the number of frames for the right context

    :return: a tuple of three values, the number of frames, the sum of frames and the sum of squares
    """
    features_server, show, start, stop, traps = input_segment
    # Load the segment of frames plus left and right context
    feat, _ = features_server.load(show,
                                   start=start-features_server.context[0],
                                   stop=stop+features_server.context[1])
    if traps:
        # Get traps
        feat, _ = features_server.get_traps(feat=feat,
                                            label=None,
                                            start=features_server.context[0],
                                            stop=feat.shape[0] - features_server.context[1])
    else:
        # Get features in context
        feat, _ = features_server.get_context(feat=feat,
                                              label=None,
                                              start=features_server.context[0],
                                              stop=feat.shape[0] - features_server.context[1])
    return feat.shape[0], feat.sum(axis=0), numpy.sum(feat ** 2, axis=0)


def mean_std_many(features_server, feature_size, seg_list, traps=False, num_thread=1):
    """
    Compute the mean and standard deviation from a list of segments.

    :param features_server: FeaturesServer used to load data
    :param feature_size: dimension o the features to accumulate
    :param seg_list: list of file names with start and stop indices
    :param traps: apply traps processing on the features in context
    :param traps: apply traps processing on the features in context
    :param num_thread: number of parallel processing to run
    :return: a tuple of three values, the number of frames, the mean and the standard deviation
    """
    inputs = [(copy.deepcopy(features_server), seg[0], seg[1], seg[2], traps) for seg in seg_list]
    pool = multiprocessing.Pool(processes=num_thread)
    res = pool.map(segment_mean_std_hdf5, inputs)

    total_n = 0
    total_f = numpy.zeros(feature_size)
    total_s = numpy.zeros(feature_size)
    for N, F, S in res:
        total_n += N
        total_f += F
        total_s += S
    return total_n, total_f / total_n, total_s / total_n



def init_weights(module):
    if type(module) == torch.nn.Linear:
        module.weight.data.normal_(0.0, 0.1)
        if module.bias is not None:
            module.bias.data.uniform_(-4.1, -3.9)


class FForwardNetwork():
    def __init__(self,
                 model,
                 filename=None,
                 input_mean=None,
                 input_std=None,
                 output_file_name=None,
                 optimizer='adam'
                 ):
        """

        """
        self.model = model
        self.input_mean = input_mean
        self.input_std = input_std
        self.optimizer = optimizer
        if output_file_name is None:
            self.output_file_name = "MyModel.mdl"
        else:
            self.output_file_name = output_file_name

    def random_init(self):
        """
        Randomly initialize the model parameters (weights and bias)
        """
        self.model.apply(init_weights)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.model.forward(x)

    def train(self,
              training_seg_list,
              cross_validation_seg_list,
              feature_size,
              segment_buffer_size=200,
              batch_size=512,
              nb_epoch=20,
              features_server_params=None,
              output_file_name="",
              traps=False,
              logger=None,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
              num_thread=2):

        # shuffle the training list
        shuffle_idx = numpy.random.permutation(numpy.arange(len(training_seg_list)))
        training_seg_list = [training_seg_list[idx] for idx in shuffle_idx]
        # split the list of files to process
        training_segment_sets = [training_seg_list[i:i + segment_buffer_size]
                                 for i in range(0, len(training_seg_list), segment_buffer_size)]

        # If not done yet, compute mean and standard deviation on all training data
        if self.input_mean is None or self.input_std is None:
            logger.critical("Compute mean and std")
            if True:
                fs = sidekit.FeaturesServer(**features_server_params)
                #self.log.info("Compute mean and standard deviation from the training features")
                feature_nb, self.input_mean, self.input_std = mean_std_many(fs,
                                                                            feature_size,
                                                                            training_seg_list,
                                                                            traps=traps,
                                                                            num_thread=num_thread)
                logger.critical("Done")
            else:
                data = numpy.load("mean_std.npz")
                self.input_mean = data["mean"]
                self.input_std = data["std"]

        # Move model to requested device (GPU)
        self.model.to(device)

        # Set training parameters
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        # Set optimizer, default is Adam
        if self.optimizer.lower() is 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())
        elif self.optimizer.lower() is 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01, momentum=0.9)
        elif self.optimizer.lower() is 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters())
        else:
            logger.critical("unknown optimizer, using default Adam")
            self.optimizer = torch.optim.Adam(self.model.parameters())

        # Initialized cross validation error
        last_cv_error = numpy.inf

        for ep in range(nb_epoch):

            logger.critical("Start epoch {} / ".format(ep, nb_epoch))
            features_server = sidekit.FeaturesServer(**features_server_params)
            running_loss = accuracy = n = nbatch = 0.0

            for idx_mb, file_list in enumerate(training_segment_sets):
                traps = False
                l = []
                f = []
                for idx, val in enumerate(file_list):
                    show, s, _, label = val
                    e = s + len(label)
                    l.append(label)
                    # Load the segment of frames plus left and right context
                    feat, _ = features_server.load(show,
                                                   start=s - features_server.context[0],
                                                   stop=e + features_server.context[1])
                    if traps:
                        # Get features in context
                        f.append(features_server.get_traps(feat=feat,
                                                           label=None,
                                                           start=features_server.context[0],
                                                           stop=feat.shape[0]-features_server.context[1])[0])
                    else:
                        # Get features in context
                        f.append(features_server.get_context(feat=feat,
                                                             label=None,
                                                             start=features_server.context[0],
                                                             stop=feat.shape[0]-features_server.context[1])[0])
                lab = numpy.hstack(l)
                fea = numpy.vstack(f).astype(numpy.float32)
                assert numpy.all(lab != -1) and len(lab) == len(fea)  # make sure that all frames have defined label
                shuffle = numpy.random.permutation(len(lab))
                label = lab.take(shuffle, axis=0)
                data = fea.take(shuffle, axis=0)

                # normalize the input
                data = (data - self.input_mean) / self.input_std

                # Send data and label to the GPU
                data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
                label = torch.from_numpy(label).to(device)

                for jj, (X, t) in enumerate(zip(torch.split(data, batch_size), torch.split(label, batch_size))):

                    self.optimizer.zero_grad()
                    lab_pred = self.forward(X)
                    loss = self.criterion(lab_pred, t)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() / (batch_size * nbatch)
                    accuracy += (torch.argmax(lab_pred.data, 1) == t).sum().item()
                    nbatch += 1
                    n += len(X)
                    if nbatch % 200 == 199:
                        logger.critical("loss = {} | accuracy = {} ".format(running_loss,  accuracy / n) )

            logger.critical("Start Cross-Validation")
            self.optimizer.zero_grad()
            running_loss = accuracy = n = nbatch = 0.0

            for ii, cv_segment in enumerate(cross_validation_seg_list):
                show, s, e, label = cv_segment
                e = s + len(label)
                t = label.astype(numpy.int16)

                # Load the segment of frames plus left and right context
                feat, _ = features_server.load(show,
                                               start=s - features_server.context[0],
                                               stop=e + features_server.context[1])
                print("taille de feat = {}".format(feat.shape))
                if traps:
                    # Get features in context
                    X = features_server.get_traps(feat=feat,
                                                  label=None,
                                                  start=features_server.context[0],
                                                  stop=feat.shape[0] - features_server.context[1])[0].astype(numpy.float32)
                else:
                    X = features_server.get_context(feat=feat,
                                                    label=None,
                                                    start=features_server.context[0],
                                                    stop=feat.shape[0] - features_server.context[1])[0].astype(numpy.float32)


                lab_pred = self.forward(X)
                loss = self.criterion(lab_pred, t)
                running_loss += loss.item() / (batch_size * nbatch)
                accuracy += (torch.argmax(lab_pred.data, 1) == t).sum().item()
                nbatch += 1
                n += len(X)
                last_cv_error = accuracy / n

            logger.critical("Cross Validation loss = {} | accuracy = {} ".format(running_loss / nbatch, accuracy / n))

            # Save the current version of the network
            torch.save(self.model.state_dict(), output_file_name.format(ep))

            # Early stopping with very basic loss criteria
            if last_cv_error <= accuracy / n:
                break