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
from multiprocessing import Pool
import numpy
import os
import time
import warnings

import sidekit.frontend
from sidekit.sidekit_io import init_logging
# from sidekit import THEANO_CONFIG
from sidekit.sidekit_wrappers import check_path_existance

# if THEANO_CONFIG == "gpu":
#     os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'
# else:
#     os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cpu,floatX=float32'

import theano
import theano.tensor as T


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2017 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def kaldi_to_hdf5(input_file_name, output_file_name):
    """
    Convert a text file containing frame alinment from Kaldi into an
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
    pool = Pool(processes=num_thread)
    res = pool.map(segment_mean_std_hdf5, inputs)

    total_n = 0
    total_f = numpy.zeros(feature_size)
    total_s = numpy.zeros(feature_size)
    for N, F, S in res:
        total_n += N
        total_f += F
        total_s += S
    return total_n, total_f / total_n, total_s / total_n


def get_params(params):
    """
    Return parameters of into a Python dictionary format

    :param params: a list of Theano shared variables
    :return: the same variables in Numpy format in a dictionary
    """
    return {p.name: p.get_value() for p in params}


def set_params(params, param_dict):
    """
    Set the parameters in a list of Theano variables from a dictionary

    :param params: dictionary to read from
    :param param_dict: list of variables in Theano format
    """
    for p_ in params:
        p_.set_value(param_dict[p_.name])


def export_params(params, param_dict):
    """
    Export network parameters into Numpy format

    :param params: dictionary of variables in Theano format
    :param param_dict: dictionary of variables in Numpy format
    """
    for k in param_dict:
        params[k.name] = k.get_value()


class FForwardNetwork(object):
    """
    Class FForwardNetwork that implement a feed-forward neural network for multiple purposes
    """

    def __init__(self, filename=None,
                 input_size=0,
                 input_mean=numpy.empty(0),
                 input_std=numpy.empty(0),
                 hidden_layer_sizes=(),
                 layers_activations=(),
                 n_classes=0
                 ):
        if filename is not None:
            # Load DNN parameters
            self.params = dict()
            _p = numpy.load(filename)
            for k, v in _p.items():
                self.params[k] = v

        elif len(layers_activations) != len(hidden_layer_sizes) + 1:
            pass

        else:  # initialize a NN with given sizes of layers and activation functions
            assert len(layers_activations) == len(hidden_layer_sizes) + 1, \
                "Mismatch between number of hidden layers and activation functions"

            sizes = (input_size,) + tuple(hidden_layer_sizes) + (n_classes,)

            self.params = {"input_mean": input_mean.astype(T.config.floatX),
                           "input_std": input_std.astype(T.config.floatX),
                           "activation_functions": layers_activations,
                           "b{}".format(len(sizes) - 1): numpy.zeros(sizes[-1]).astype(T.config.floatX),
                           "hidden_layer_sizes": hidden_layer_sizes
                           }

            for ii in range(1, len(sizes)):
                self.params["W{}".format(ii)] = numpy.random.randn(
                        sizes[ii - 1],
                        sizes[ii]).astype(T.config.floatX) * 0.1
                self.params["b{}".format(ii)] = numpy.random.random(sizes[ii]).astype(T.config.floatX) / 5.0 - 4.1
        
        init_logging()
        self.log = logging.getLogger()

    @check_path_existance
    def write(self, output_filename):
        """
        Write a feed-forward neural network to disk in HDF5 format
        :param output_filename: the name of the output file
        """
        with h5py.File(output_filename, "w") as fh:

            # Get the number of hidden layers
            layer_number = len(self.params["hidden_layer_sizes"])

            # Write input mean and std
            fh.create_dataset("input_mean",
                              data=self.params["input_mean"],
                              compression="gzip",
                              fletcher32=True)
            fh.create_dataset("input_std",
                              data=self.params["input_std"],
                              compression="gzip",
                              fletcher32=True)

            # Write sizes of hidden layers
            fh.create_dataset("hidden_layer_sizes",
                              data=numpy.array(self.params["hidden_layer_sizes"]),
                              compression="gzip",
                              fletcher32=True)

            # Write activation functions
            tmp_activations = numpy.array(copy.deepcopy(self.params["activation_functions"]))
            activation_is_none = [act is None for act in self.params["activation_functions"]]
            numpy.place(tmp_activations, activation_is_none, "None")
            fh.create_dataset("activation_functions",
                              data=tmp_activations.astype('S'),
                              compression="gzip",
                              fletcher32=True)

            # For each layer, save biais and weights
            for layer in range(1, layer_number + 2):
                fh.create_dataset("b{}".format(layer),
                                  data=self.params["b{}".format(layer)],
                                  compression="gzip",
                                  fletcher32=True)
                fh.create_dataset("W{}".format(layer),
                                  data=self.params["W{}".format(layer)],
                                  compression="gzip",
                                  fletcher32=True)

    @staticmethod
    def read(input_filename):
        """

        :param input_filename:
        :return:
        """
        nn = FForwardNetwork()

        nn.params = dict()
        with h5py.File(input_filename, "r") as fh:
            # read input_mean and input_std
            nn.params["input_mean"] = fh["input_mean"].value
            nn.params["input_std"] = fh["input_std"].value

            # read sizes of hidden layers
            nn.params["hidden_layer_sizes"] = tuple(fh["hidden_layer_sizes"].value)
            layer_number = len(nn.params["hidden_layer_sizes"])

            # read activation functions
            tmp = fh["activation_functions"].value.astype('U255', copy=False)
            nn.params["activation_functions"] = []
            for idx, act in enumerate(tmp):
                if act == 'None':
                    nn.params["activation_functions"].append(None)
                else:
                    nn.params["activation_functions"].append(act)
            nn.params["activation_functions"] = tuple(nn.params["activation_functions"])

            # For each layer, read biais and weights
            for layer in range(1, layer_number + 2):
                nn.params["b{}".format(layer)] = fh["b{}".format(layer)].value
                nn.params["W{}".format(layer)] = fh["W{}".format(layer)].value

        init_logging()
        nn.log = logging.getLogger()
        return nn

    def replace_layer(self, layer_number, hidden_unit_number, activation_function=None):
        """

        :param layer_number:
        :param hidden_unit_number:
        :param activation_function:
        :return:
        """
        # Modify the activation function
        self.params['activation_functions'][layer_number-1] = activation_function

        # Modify the weight matrices and bias vectors before and after the modified layer
        self.params["W{}".format(layer_number)] = numpy.random.randn(
            self.params["W{}".format(layer_number)].shape[0],
            hidden_unit_number).astype(T.config.floatX) * 0.1

        self.params["W{}".format(layer_number+1)] = numpy.random.randn(
            hidden_unit_number,
            self.params["W{}".format(layer_number+1)].shape[1]).astype(T.config.floatX) * 0.1

        self.params["b{}".format(layer_number)] = numpy.random.random(hidden_unit_number).astype(T.config.floatX) \
                                                  / 5.0 - 4.1

    def instantiate_network(self):
        """ Create Theano variables and initialize the weights and biases 
        of the neural network
        Create the different funtions required to train the NN
        """

        # Define the variable for inputs
        X_ = T.matrix("X")

        # Define variables for mean and standard deviation of the input
        mean_ = theano.shared(self.params['input_mean'].astype(T.config.floatX), name='input_mean')
        std_ = theano.shared(self.params['input_std'].astype(T.config.floatX), name='input_std')

        # Define the variable for standardized inputs
        Y_ = (X_ - mean_) / std_

        # Get the list of activation functions for each layer
        activation_functions = []
        for af in self.params["activation_functions"]:
            if af == "sigmoid":
                activation_functions.append(T.nnet.sigmoid)
            elif af == "relu":
                activation_functions.append(T.nnet.relu)
            elif af == "softmax":
                activation_functions.append(T.nnet.softmax)
            elif af == "binary_crossentropy":
                activation_functions.append(T.nnet.binary_crossentropy)
            elif af is None:
                activation_functions.append(None)

        # Define list of variables 
        params_ = [mean_, std_]

        # For each layer, initialized the weights and biases
        for ii, f in enumerate(activation_functions):
            W_name = "W{}".format(ii + 1)
            b_name = "b{}".format(ii + 1)
            W_ = theano.shared(self.params[W_name].astype(T.config.floatX), name=W_name)
            b_ = theano.shared(self.params[b_name].astype(T.config.floatX), name=b_name)
            if f is None:
                Y_ = Y_.dot(W_) + b_
            else:
                Y_ = f(Y_.dot(W_) + b_)
            params_ += [W_, b_]

        return X_, Y_, params_

    def _train_acoustic(self,
               output_accuracy_limit,
               training_seg_list,
               cross_validation_seg_list,
               features_server,
               lr=0.008,
               segment_buffer_size=200,
               batch_size=512,
               max_iters=20,
               tolerance=0.003,
               output_file_name="",
               save_tmp_nnet=False,
               traps=False):
        """
        train the network and return the parameters
        Exit at the end of the training process or as soon as the output_accuracy_limit is reach on
        the training data

        Return a dictionary of the network parameters

        :param training_seg_list: list of segments to use for training
            It is a list of 4 dimensional tuples which
            first argument is the absolute file name
            second argument is the index of the first frame of the segment
            third argument is the index of the last frame of the segment
            and fourth argument is a numpy array of integer,
            labels corresponding to each frame of the segment
        :param cross_validation_seg_list: is a list of segments to use for
            cross validation. Same format as train_seg_list
        :param features_server: FeaturesServer used to load data
        :param lr: initial learning rate
        :param segment_buffer_size: number of segments loaded at once
        :param batch_size: size of the minibatches as number of frames
        :param max_iters: macimum number of epochs
        :param tolerance:
        :param output_file_name: root name of the files to save Neural Betwork parameters
        :param save_tmp_nnet: boolean, if True, save the parameters after each epoch
        :param traps: boolean, if True, compute TRAPS on the input data, if False jsut use concatenated frames
        :return:
        """
        numpy.random.seed(42)

        # Instantiate the neural network, variables used to define the network
        # are defined and initialized
        X_, Y_, params_ = self.instantiate_network()

        # define a variable for the learning rate
        lr_ = T.scalar()

        # Define a variable for the output labels
        T_ = T.ivector("T")

        # Define the functions used to train the network
        cost_ = T.nnet.categorical_crossentropy(Y_, T_).sum()
        acc_ = T.eq(T.argmax(Y_, axis=1), T_).sum()
        params_to_update_ = [p for p in params_ if p.name[0] in "Wb"]
        grads_ = T.grad(cost_, params_to_update_)

        train = theano.function(
                inputs=[X_, T_, lr_],
                outputs=[cost_, acc_],
                updates=[(p, p - lr_ * g) for p, g in zip(params_to_update_, grads_)])

        xentropy = theano.function(inputs=[X_, T_], outputs=[cost_, acc_])

        # split the list of files to process
        training_segment_sets = [training_seg_list[i:i + segment_buffer_size]
                                 for i in range(0, len(training_seg_list), segment_buffer_size)]

        # Initialized cross validation error
        last_cv_error = numpy.inf

        # Set the initial decay factor for the learning rate
        lr_decay_factor = 1

        # Iterate to train the network
        for kk in range(1, max_iters):
            lr *= lr_decay_factor  # update the learning rate

            error = accuracy = n = 0.0
            nfiles = 0

            # Iterate on the mini-batches
            for ii, training_segment_set in enumerate(training_segment_sets):
                start_time = time.time()
                l = []
                f = []
                for idx, val in enumerate(training_segment_set):
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

                lab = numpy.hstack(l).astype(numpy.int16)
                fea = numpy.vstack(f).astype(numpy.float32)
                assert numpy.all(lab != -1) and len(lab) == len(fea)  # make sure that all frames have defined label
                shuffle = numpy.random.permutation(len(lab))
                lab = lab.take(shuffle, axis=0)
                fea = fea.take(shuffle, axis=0)

                nsplits = len(fea) / batch_size
                nfiles += len(training_segment_set)

                for jj, (X, t) in enumerate(zip(numpy.array_split(fea, nsplits), numpy.array_split(lab, nsplits))):
                    err, acc = train(X.astype(numpy.float32), t.astype(numpy.int16), lr)
                    error += err
                    accuracy += acc
                    n += len(X)
                self.log.info("%d/%d | %f | %f ", nfiles, len(training_seg_list), error / n, accuracy / n)
                self.log.info("time = {}".format(time.time() - start_time))

                # Exit if the output_accuracy_limit has been reached
                print("accuracy = {}, output_accuracy_limit = {}".format(100*accuracy/n, output_accuracy_limit))
                if 100*accuracy/n >= output_accuracy_limit:
                    tmp_dict = get_params(params_)
                    tmp_dict.update({"hidden_layer_sizes": self.params["hidden_layer_sizes"]})
                    tmp_dict.update({"activation_functions": self.params["activation_functions"]})
                    return tmp_dict

            error = accuracy = n = 0.0

            # Cross-validation
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

                assert len(X) == len(t)
                err, acc = xentropy(X, t)
                error += err
                accuracy += acc
                n += len(X)


            # Save the current version of the network
            if save_tmp_nnet:
                tmp_hls = self.params["hidden_layer_sizes"]
                tmp_af = self.params["activation_functions"]
                self.params = get_params(params_)
                self.params["hidden_layer_sizes"] = tmp_hls
                self.params["activation_functions"] = tmp_af
                self.write(output_file_name + '_epoch_{}'.format(kk))

            # Load previous weights if error increased
            if last_cv_error <= error:
                set_params(params_, last_params)
                error = last_cv_error

            # Start halving the learning rate or terminate the training
            if (last_cv_error - error) / numpy.abs([last_cv_error, error]).max() <= tolerance:
                if lr_decay_factor < 1:
                    break
                lr_decay_factor = 0.5

            # Update the cross-validation error
            last_cv_error = error

            # get last computed params
            last_params = get_params(params_)

        # Return the last parameters
        tmp_dict = get_params(params_)
        tmp_dict.update({"hidden_layer_sizes": self.params["hidden_layer_sizes"]})
        tmp_dict.update({"activation_functions": self.params["activation_functions"]})
        return tmp_dict

    def train_acoustic(self,
                       training_seg_list,
                       cross_validation_seg_list,
                       features_server,
                       feature_size,
                       lr=0.008,
                       segment_buffer_size=200,
                       batch_size=512,
                       max_iters=20,
                       tolerance=0.003,
                       output_file_name="",
                       save_tmp_nnet=False,
                       traps=False,
                       num_thread=1):
        """

        :param training_seg_list: list of segments to use for training
            It is a list of 4 dimensional tuples which
            first argument is the absolute file name
            second argument is the index of the first frame of the segment
            third argument is the index of the last frame of the segment
            and fourth argument is a numpy array of integer,
            labels corresponding to each frame of the segment
        :param cross_validation_seg_list: is a list of segments to use for
            cross validation. Same format as train_seg_list
        :param features_server: FeaturesServer used to load data
        :param feature_size: dimension of the acoustic feature
        :param lr: initial learning rate
        :param segment_buffer_size: number of segments loaded at once
        :param batch_size: size of the minibatches as number of frames
        :param max_iters: macimum number of epochs
        :param tolerance:
        :param output_file_name: root name of the files to save Neural Betwork parameters
        :param save_tmp_nnet: boolean, if True, save the parameters after each epoch
        :param traps: boolean, if True, compute TRAPS on the input data, if False jsut use concatenated frames
        :param num_thread: number of parallel process to run (for CPU part of the code)
        :return:
        """
        numpy.random.seed(42)

        # shuffle the training list
        shuffle_idx = numpy.random.permutation(numpy.arange(len(training_seg_list)))
        training_seg_list = [training_seg_list[idx] for idx in shuffle_idx]

        # If not done yet, compute mean and standard deviation on all training data
        if 0 in [len(self.params["input_mean"]), len(self.params["input_std"])]:

            if True:
                self.log.info("Compute mean and standard deviation from the training features")
                feature_nb, self.params["input_mean"], self.params["input_std"] = mean_std_many(features_server,
                                                                                                feature_size,
                                                                                                training_seg_list,
                                                                                                traps=traps,
                                                                                                num_thread=num_thread)
                sidekit.sidekit_io.write_dict_hdf5({"input_mean_std": self.params["input_mean"],
                                                    "input_std": self.params["input_std"]}, output_file_name + '_final')

            else:
                self.log.info("Load input mean and standard deviation from file")
                self.params["input_mean"] = numpy.zeros(360)
                self.params["input_std"] = numpy.ones(360)

        # Train the model and get the parameters
        self.params = self._train_acoustic(numpy.inf,
                                           training_seg_list,
                                           cross_validation_seg_list,
                                           features_server,
                                           lr,
                                           segment_buffer_size,
                                           batch_size,
                                           max_iters,
                                           tolerance,
                                           output_file_name,
                                           save_tmp_nnet,
                                           traps)

        # Save final network
        self.write(output_file_name + '_final')

    def instantiate_partial_network(self, layer_number):
        """
        Instantiate a neural network with only the bottom layers of the network.
        After instantiating, the function display the structure of the network in the root logger if it exists
        :param layer_number: number of layers to load from
        """
        # Define the variable for inputs
        X_ = T.matrix("X")

        # Define variables for mean and standard deviation of the input
        mean_ = theano.shared(self.params['input_mean'].astype(T.config.floatX), name='input_mean')
        std_ = theano.shared(self.params['input_std'].astype(T.config.floatX), name='input_std')

        # Define the variable for standardized inputs
        Y_ = (X_ - mean_) / std_

        # Get the list of activation functions for each layer
        activation_functions = []
        for af in self.params["activation_functions"][:layer_number]:
            
            if af == "sigmoid":
                activation_functions.append(T.nnet.sigmoid)
            elif af == "relu":
                activation_functions.append(T.nnet.relu)
            elif af == "softmax":
                activation_functions.append(T.nnet.softmax)
            elif af == "binary_crossentropy":
                activation_functions.append(T.nnet.binary_crossentropy)
            elif af is None:
                activation_functions.append(None)
            
        # Define list of variables
        params_ = [mean_, std_]

        # For each layer, initialized the weights and biases
        for ii, f in enumerate(activation_functions):
            W_name = "W{}".format(ii + 1)
            b_name = "b{}".format(ii + 1)
            W_ = theano.shared(self.params[W_name].astype(T.config.floatX), name=W_name)
            b_ = theano.shared(self.params[b_name].astype(T.config.floatX), name=b_name)
            if f is None:
                Y_ = Y_.dot(W_) + b_
            else:
                Y_ = f(Y_.dot(W_) + b_)
            params_ += [W_, b_]

        return X_, Y_, params_

    def train_per_layer(self,
                        layer_training_sequence,  # tuple: number of layers to add at each step
                        training_accuracy_limit,  # tuple: accuracy to target for each step, once reached
                        training_seg_list,
                        cross_validation_seg_list,
                        features_server,
                        feature_size,
                        lr=0.008,
                        segment_buffer_size=200,
                        batch_size=512,
                        max_iters=20,
                        tolerance=0.003,
                        output_file_name="",
                        save_tmp_nnet=False,
                        traps=False,
                        num_thread=1):
        """

        :param layer_training_sequence:
        :param training_accuracy_limit:
        :param training_seg_list:
        :param cross_validation_seg_list:
        :param features_server:
        :param feature_size:
        :param lr:
        :param segment_buffer_size:
        :param batch_size:
        :param max_iters:
        :param tolerance:
        :param output_file_name:
        :param save_tmp_nnet:
        :param traps:
        :param num_thread:
        :return:
        """
        numpy.random.seed(42)

        # shuffle the training list
        shuffle_idx = numpy.random.permutation(numpy.arange(len(training_seg_list)))
        training_seg_list = [training_seg_list[idx] for idx in shuffle_idx]

        # If not done yet, compute mean and standard deviation on all training data
        if 0 in [len(self.params["input_mean"]), len(self.params["input_std"])]:

            if True:
                self.log.info("Compute mean and standard deviation from the training features")
                feature_nb, self.params["input_mean"], self.params["input_std"] = mean_std_many(features_server,
                                                                                                feature_size,
                                                                                                training_seg_list,
                                                                                                traps=traps,
                                                                                                num_thread=num_thread)
                sidekit.sidekit_io.write_dict_hdf5({"input_mean_std": self.params["input_mean"],
                                                    "input_std": self.params["input_std"]}, output_file_name + '_final')

            else:
                self.log.info("Load input mean and standard deviation from file")
                ms = numpy.load("input_mean_std.npz")
                self.params["input_mean"] = ms["input_mean"]
                self.params["input_std"] = ms["input_std"]

        n_classes = self.params["b{}".format(len(self.params["activation_functions"]))].shape[0]

        tmp_nn = sidekit.theano_utils.FForwardNetwork(input_size=feature_size,
            hidden_layer_sizes=tuple([self.params["hidden_layer_sizes"][ii]
                                      for ii in range(layer_training_sequence[0])]),
            layers_activations=tuple([self.params["activation_functions"][ii]
                                      for ii in range(layer_training_sequence[0])]) + ("softmax",), n_classes=n_classes)
        tmp_nn.params["input_mean"] = self.params["input_mean"]
        tmp_nn.params["input_std"] = self.params["input_std"]

        init_params = tmp_nn._train_acoustic(training_accuracy_limit[0],
                                             training_seg_list,
                                             cross_validation_seg_list,
                                             features_server,
                                             lr,
                                             segment_buffer_size,
                                             batch_size,
                                             max_iters,
                                             tolerance,
                                             output_file_name,
                                             save_tmp_nnet,
                                             traps)
        """ Pour chaque couche (ou groupe de couche)  """
        for iteration in range(1, len(layer_training_sequence)):

            previous_layer_number = numpy.cumsum(layer_training_sequence)[iteration - 1]
            new_layer_number = numpy.cumsum(layer_training_sequence)[iteration]
            #
            init_params["activation_functions"] = tuple(self.params["activation_functions"]
                                                        [:new_layer_number]) + ("softmax",)
            init_params["hidden_layer_sizes"] = self.params["hidden_layer_sizes"][:new_layer_number]
            sizes = init_params["hidden_layer_sizes"] + (n_classes,)
            #
            for layer in range(previous_layer_number, new_layer_number + 1):
                # Modify the previous last layer biais (re-initialize to enter that new layer)
                init_params["b{}".format(layer + 1)] = \
                    numpy.random.random(sizes[layer]).astype(T.config.floatX) / 5.0 - 4.1
                #
                init_params["W{}".format(layer + 1)] = numpy.random.randn(
                    sizes[layer - 1],
                    sizes[layer]).astype(T.config.floatX) * 0.1

            tmp_nn.params = init_params
            init_params = tmp_nn._train_acoustic(training_accuracy_limit[iteration],
                                                 training_seg_list,
                                                 cross_validation_seg_list,
                                                 features_server,
                                                 lr,
                                                 segment_buffer_size,
                                                 batch_size,
                                                 max_iters,
                                                 tolerance,
                                                 output_file_name,
                                                 save_tmp_nnet,
                                                 traps)

    def feed_forward_acoustic(self,
                              feature_file_list,
                              features_server,
                              layer_number,
                              output_file_structure):
        """
        Function used to extract bottleneck features or embeddings from an existing Neural Network.
        The first bottom layers of the neural network are loaded and all feature files are process through
        the network to get the output and save them as feature files.
        If specified, the output features can be normalized (cms, cmvn, stg) given input labels

        :param feature_file_list: list of feature files to process through the feed formward network
        :param features_server: FeaturesServer used to load the data
        :param layer_number: number of layers to load from the model
        :param output_file_structure: structure of the output file name
        :return:
        """
        # Instantiate the network
        X_, Y_, params_ = self.instantiate_partial_network(layer_number)

        # Define the forward function to get the output of the first network: bottle-neck features
        forward = theano.function(inputs=[X_], outputs=Y_)

        for show in feature_file_list:
            self.log.info("Process file %s", show)

            # Load the segment of frames plus left and right context
            feat, label = features_server.load(show)
            # Get bottle neck features from features in context
            bnf = forward(features_server.get_context(feat=feat)[0])

            # Create the directory if it doesn't exist
            dir_name = os.path.dirname(output_file_structure.format(show))  # get the path
            if not os.path.exists(dir_name) and (dir_name is not ''):
                os.makedirs(dir_name)

            # Save in HDF5 format, labels are saved if they don't exist in thge output file
            with h5py.File(output_file_structure.format(show), "a") as h5f:
                vad = None if show + "vad" in h5f else label
                bnf_mean = bnf[vad, :].mean(axis=0)
                bnf_std = bnf[vad, :].std(axis=0)
                sidekit.frontend.io.write_hdf5(show, h5f, 
                                               None, None, None, 
                                               None, None, None, 
                                               None, None, None, 
                                               bnf, bnf_mean, bnf_std,
                                               vad)

    def _train(self,
               output_accuracy_limit,
               training_set,
               cross_validation_set,
               lr=0.008,
               batch_size=512,
               max_iters=20,
               tolerance=0.003,
               output_file_name="",
               save_tmp_nnet=False):
        """
        train the network and return the parameters
        Exit at the end of the training process or as soon as the output_accuracy_limit is reach on
        the training data

        Return a dictionary of the network parameters

        :param training_seg_list: list of segments to use for training
            It is a list of 4 dimensional tuples which
            first argument is the absolute file name
            second argument is the index of the first frame of the segment
            third argument is the index of the last frame of the segment
            and fourth argument is a numpy array of integer,
            labels corresponding to each frame of the segment
        :param cross_validation_seg_list: is a list of segments to use for
            cross validation. Same format as train_seg_list
        :param features_server: FeaturesServer used to load data
        :param lr: initial learning rate
        :param segment_buffer_size: number of segments loaded at once
        :param batch_size: size of the minibatches as number of frames
        :param max_iters: macimum number of epochs
        :param tolerance:
        :param output_file_name: root name of the files to save Neural Betwork parameters
        :param save_tmp_nnet: boolean, if True, save the parameters after each epoch
        :param traps: boolean, if True, compute TRAPS on the input data, if False jsut use concatenated frames
        :return:
        """
        numpy.random.seed(42)

        # Instantiate the neural network, variables used to define the network
        # are defined and initialized
        X_, Y_, params_ = self.instantiate_network()

        # define a variable for the learning rate
        lr_ = T.scalar()

        # Define a variable for the output labels
        T_ = T.ivector("T")

        # Define the functions used to train the network
        cost_ = T.nnet.categorical_crossentropy(Y_, T_).sum()
        acc_ = T.eq(T.argmax(Y_, axis=1), T_).sum()
        params_to_update_ = [p for p in params_ if p.name[0] in "Wb"]
        grads_ = T.grad(cost_, params_to_update_)

        train = theano.function(
                inputs=[X_, T_, lr_],
                outputs=[cost_, acc_],
                updates=[(p, p - lr_ * g) for p, g in zip(params_to_update_, grads_)])

        xentropy = theano.function(inputs=[X_, T_], outputs=[cost_, acc_])

        # Initialized cross validation error
        last_cv_error = numpy.inf

        # Set the initial decay factor for the learning rate
        lr_decay_factor = 1

        # Iterate to train the network
        for kk in range(1, max_iters):
            lr *= lr_decay_factor  # update the learning rate

            error = accuracy = n = 0.0
            nfiles = 0

            # Iterate on the mini-batches
            nsplits = len(training_set[0]) / batch_size
            for jj, (X, t) in enumerate(zip(numpy.array_split(training_set[0], nsplits),
                                            numpy.array_split(training_set[1], nsplits))):
                err, acc = train(X.astype(numpy.float32), t.astype(numpy.int16), lr)
                error += err
                accuracy += acc
                n += len(X)
                nfiles += len(t)
            self.log.info("%d/%d | %f | %f ", nfiles, len(training_set[1]), error / n, accuracy / n)

            # Exit if the output_accuracy_limit has been reached
            if 100*accuracy/n >= output_accuracy_limit:
                tmp_dict = get_params(params_)
                tmp_dict.update({"hidden_layer_sizes": self.params["hidden_layer_sizes"]})
                tmp_dict.update({"activation_functions": self.params["activation_functions"]})
                return tmp_dict

            error = accuracy = n = 0.0

            # Cross-validation
            nsplits = len(cross_validation_set[0]) / batch_size
            for jj, (X, t) in enumerate(zip(numpy.array_split(cross_validation_set[0], nsplits),
                                            numpy.array_split(cross_validation_set[1], nsplits))):
                assert len(X) == len(t)
                err, acc = xentropy(X, t)
                error += err
                accuracy += acc
                n += len(X)
            self.log.info("Cross-validation | %f | %f ", error / n, accuracy / n)

            # Save the current version of the network
            if save_tmp_nnet:
                tmp_dict = get_params(params_)
                tmp_dict.update({"hidden_layer_sizes": self.params["hidden_layer_sizes"]})
                tmp_dict.update({"activation_functions": self.params["activation_functions"]})
                self.write(output_file_name + '_epoch_{}'.format(kk))

            # Load previous weights if error increased
            if last_cv_error <= error:
                set_params(params_, last_params)
                error = last_cv_error

            # Start halving the learning rate or terminate the training
            if (last_cv_error - error) / numpy.abs([last_cv_error, error]).max() <= tolerance:
                print("reduced lr")
                if lr_decay_factor < 1:
                    break
                lr_decay_factor = 0.5

            # Update the cross-validation error
            last_cv_error = error

            # get last computed params
            last_params = get_params(params_)

        # Return the last parameters
        tmp_dict = get_params(params_)
        tmp_dict.update({"hidden_layer_sizes": self.params["hidden_layer_sizes"]})
        tmp_dict.update({"activation_functions": self.params["activation_functions"]})
        return tmp_dict

    def train(self,
              training_set,
              cross_validation_set,
              lr=0.008,
              batch_size=512,
              max_iters=20,
              tolerance=0.003,
              output_file_name="",
              save_tmp_nnet=False,
              num_thread=1):
        """

        :param training_set: list of segments to use for training
            It is a list of 4 dimensional tuples which
            first argument is the absolute file name
            second argument is the index of the first frame of the segment
            third argument is the index of the last frame of the segment
            and fourth argument is a numpy array of integer,
            labels corresponding to each frame of the segment
        :param cross_validation_set: is a list of segments to use for
            cross validation. Same format as train_seg_list
        :param lr: initial learning rate
        :param batch_size: size of the minibatches as number of frames
        :param max_iters: macimum number of epochs
        :param tolerance:
        :param output_file_name: root name of the files to save Neural Network parameters
        :param save_tmp_nnet: boolean, if True, save the parameters after each epoch
        :param num_thread: number of parallel process to run (for CPU part of the code)
        :return:
        """
        numpy.random.seed(42)

        # shuffle the training set
        shuffle_idx = numpy.random.permutation(numpy.arange(len(training_set[0])))
        training_set[0] = training_set[0][shuffle_idx, :]  # shuffle the data
        training_set[1] = training_set[1][shuffle_idx]  # shuffle the labels

        # Compute mean and standard deviation on all training data
        self.params["input_mean"] = training_set[0].mean(axis=0)
        self.params["input_std"] = training_set[0].std(axis=0)

        # Train the model and get the parameters
        self.params = self._train(numpy.inf,
                                  training_set,
                                  cross_validation_set,
                                  lr,
                                  batch_size,
                                  max_iters,
                                  tolerance,
                                  output_file_name,
                                  save_tmp_nnet)

        # Save final network
        self.write(output_file_name + '_final')

    def feed_forward(self,
                     data,
                     layer_number):
        """
        Function used to extract bottleneck features or embeddings from an existing Neural Network.
        The first bottom layers of the neural network are loaded and all feature files are process through
        the network to get the output and save them as feature files.
        If specified, the output features can be normalized (cms, cmvn, stg) given input labels

        :param data: data to process (a Numpy array)
        :param layer_number: number of layers to load from the model

        :return: the transform data
        """
        # Instantiate the network
        X_, Y_, params_ = self.instantiate_partial_network(layer_number)

        # Define the forward function to get the output of the first network: bottle-neck features
        forward = theano.function(inputs=[X_], outputs=Y_)

        # Process the input data through the DNN
        return forward(data.astype(numpy.float32))

    def compute_ubm_dnn(self,
                        training_list,
                        dnn_features_server,
                        features_server,
                        viterbi=False):
        """

        :param training_list: list of files to process to train the model
        :param dnn_features_server: FeaturesServer to feed the network
        :param features_server: FeaturesServer providing features to compute the first and second order statistics
        :param viterbi: boolean, if True, keep only one coefficient to one and the others at zeros
        :return: a Mixture object
        """

        os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'
        # Accumulate statistics using the DNN (equivalent to E step)

        # Load weight parameters and create a network
        X_, Y_, params_ = self.instantiate_network()

        ndim = self.params["b{}".format(len(self.params['activation_functions']))].shape[0]  # number of distributions

        print("Train a UBM with {} Gaussian distributions".format(ndim))

        # Define the forward function to get the output of the network
        forward = theano.function(inputs=[X_], outputs=Y_)

        # Initialize the accumulator given the size of the first feature file
        feature_size = features_server.load(training_list[0])[0].shape[1]

        # Initialize one Mixture for UBM storage and one Mixture to accumulate the
        # statistics
        ubm = sidekit.Mixture()
        ubm.cov_var_ctl = numpy.ones((ndim, feature_size))

        accum = sidekit.Mixture()
        accum.mu = numpy.zeros((ndim, feature_size))
        accum.invcov = numpy.zeros((ndim, feature_size))
        accum.w = numpy.zeros(ndim)

        # Compute the zero, first and second order statistics
        for idx, seg in enumerate(training_list):

            print("accumulate stats: {}".format(seg))
            # Process the current segment and get the stat0 per frame
            features, _ = dnn_features_server.load(seg)
            stat_features, labels = features_server.load(seg)
            
            s0 = forward(dnn_features_server.get_context(feat=features)[0])[labels]
            stat_features = stat_features[labels, :]

            if viterbi:
                max_idx = s0.argmax(axis=1)
                z = numpy.zeros(s0.shape).flatten()
                z[numpy.ravel_multi_index(numpy.vstack((numpy.arange(30), max_idx)), s0.shape)] = 1.
                s0 = z.reshape(s0.shape)

            # zero order statistics
            accum.w += s0.sum(0)

            # first order statistics
            accum.mu += numpy.dot(stat_features.T, s0).T

            # second order statistics
            accum.invcov += numpy.dot(numpy.square(stat_features.T), s0).T

        # M step
        ubm._maximization(accum)

        return ubm


def compute_stat_dnn_parallel(feed_forward_nnet,
                              segset,
                              stat0,
                              stat1,
                              dnn_features_server,
                              features_server,
                              seg_indices=None):
    """
    Single thread version of the statistic computation using a DNN.

    :param feed_forward_nnet: weights and biaises of the network stored in npz format
    :param segset: list of segments to process
    :param stat0: local matrix of zero-order statistics
    :param stat1: local matrix of first-order statistics
    :param dnn_features_server: FeaturesServer that provides input data for the DNN
    :param features_server: FeaturesServer that provide additional features to compute first order statistics
    :param seg_indices: indices of the
    :return: a StatServer with all computed statistics
    """

    """
    :param nn_file_name:
    :param idmap: class name, session name and start/ stop information
       of each segment to process in an IdMap object

    :return: a StatServer...
    """
    FfNn = sidekit.nnet.FForwardNetwork.read(feed_forward_nnet)

    # Load weight parameters and create a network
    X_, Y_, params_ = FfNn.instantiate_network()

    # Define the forward function to get the output of the network
    forward = theano.function(inputs=[X_], outputs=Y_)

    for idx in seg_indices:
        print("Compute statistics for {}".format(segset[idx]))
        logging.debug('Compute statistics for {}'.format(segset[idx]))

        show = segset[idx]
        channel = 0
        if features_server.features_extractor is not None \
                and show.endswith(features_server.double_channel_extension[1]):
            channel = 1
        stat_features, labels = features_server.load(show, channel=channel)
        features, _ = dnn_features_server.load(show, channel=channel)
        stat_features = stat_features[labels, :]

        s0 = forward(dnn_features_server.get_context(feat=features)[0])[labels]
        s1 = numpy.dot(stat_features.T, s0).T

        stat0[idx, :] = s0.sum(axis=0)
        stat1[idx, :] = s1.flatten()


def compute_stat_dnn(idmap,
                     feed_forward_nnet,
                     dnn_features_server,
                     features_server,
                     num_thread=1):
    """

    :param idmap: IdMap that describes segment to process
    :param feed_forward_nnet: neural network to load
    :param dnn_features_server: FeaturesServer to feed the Neural Network
    :param features_server: FeaturesServer that provide additional features to compute first order statistics
    :param num_thread: number of parallel process to run
    :return:
    """
    # Get the number of distributions
    FfNn = sidekit.nnet.FForwardNetwork.read(feed_forward_nnet)
    ndim = FfNn.params["b{}".format(len(FfNn.params['activation_functions']))].shape[0]

    # get dimension of the features
    feature_size = features_server.load(idmap.rightids[0])[0].shape[1]

    # Create and initialize a StatServer
    ss = sidekit.StatServer(idmap)
    ss.stat0 = numpy.zeros((idmap.leftids.shape[0], ndim), dtype=numpy.float32)
    ss.stat1 = numpy.zeros((idmap.leftids.shape[0], ndim * feature_size), dtype=numpy.float32)

    with warnings.catch_warnings():
        ct = ctypes.c_float
        warnings.simplefilter('ignore', RuntimeWarning)
        tmp_stat0 = multiprocessing.Array(ct, ss.stat0.size)
        ss.stat0 = numpy.ctypeslib.as_array(tmp_stat0.get_obj())
        ss.stat0 = ss.stat0.reshape(ss.segset.shape[0], ndim)

        tmp_stat1 = multiprocessing.Array(ct, ss.stat1.size)
        ss.stat1 = numpy.ctypeslib.as_array(tmp_stat1.get_obj())
        ss.stat1 = ss.stat1.reshape(ss.segset.shape[0], ndim * feature_size)

    # Split indices
    sub_lists = numpy.array_split(numpy.arange(idmap.leftids.shape[0]), num_thread)

    # Start parallel processing (make sure THEANO uses CPUs)
    jobs = []
    multiprocessing.freeze_support()
    for idx in range(num_thread):
        p = multiprocessing.Process(target=compute_stat_dnn_parallel, 
                                    args=(feed_forward_nnet,
                                          ss.segset,
                                          ss.stat0,
                                          ss.stat1,
                                          copy.deepcopy(dnn_features_server),
                                          copy.deepcopy(features_server),
                                          sub_lists[idx]))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
        
    # Return StatServer
    return ss

