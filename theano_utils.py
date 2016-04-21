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
Copyright 2014-2016 Anthony Larcher

:mod:`theano_utils` provides utilities to facilitate the work with SIDEKIT
and THEANO.

The authors would like to thank the BUT Speech@FIT group (http://speech.fit.vutbr.cz) and Lukas BURGET
for sharing the source code that strongly inspired this module. Thank you for your valuable contribution.
"""
import numpy as np
import os
import logging
from multiprocessing import Pool

import sidekit.frontend
from sidekit.sidekit_io import init_logging
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cpu,floatX=float32'
import theano
import theano.tensor as T


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def segment_mean_std_spro4(input_segment):
    """
    Compute the sum and square sum of all features for a list of segments.
    Input files are in SPRO4 format

    :param input_segment: list of segments to read from, each element of the list is a tuple of 5 values, the filename, the index of the first frame, index of the last frame, the number of frames for the left context and the number of frames for the right context

    :return: a tuple of three values, the number of frames, the sum of frames and the sum of squares
    """
    filename, start, stop, left_context, right_context = input_segment
    feat = sidekit.frontend.features.get_context(
            sidekit.frontend.io.read_spro4_segment(filename,
                                                   start=start,
                                                   end=stop),
            left_ctx=left_context,
            right_ctx=right_context,
            apply_hamming=False)
    return feat.shape[0], feat.sum(axis=0), np.sum(feat ** 2, axis=0)


def segment_mean_std_htk(input_segment):
    """
    Compute the sum and square sum of all features for a list of segments.
    Input files are in HTK format

    :param input_segment: list of segments to read from, each element of the list is a tuple of 5 values, the filename, the index of thefirst frame, index of the last frame, the number of frames for the left context and the number of frames for the right context

    :return: a tuple of three values, the number of frames, the sum of frames and the sum of squares
    """
    filename, start, stop, left_context, right_context = input_segment
    feat = sidekit.frontend.features.get_context(
            sidekit.frontend.io.read_htk_segment(filename,
                                                 start=start,
                                                 end=stop),
            left_ctx=left_context,
            right_ctx=right_context,
            apply_hamming=False)
    return feat.shape[0], feat.sum(axis=0), np.sum(feat ** 2, axis=0)


def mean_std_many(file_format, feature_size, seg_list, left_context, right_context):
    """
    Compute the mean and standard deviation from a list of segments.

    :param file_format: should be 'spro4' or 'htk'
    :param feature_size: dimension o the features to accumulate
    :param seg_list: list of file names with start and stop indices
    :param left_context: number of frames to add for the left context
    :param right_context: number of frames to add for the right context

    :return: a tuple of three values, the number of frames, the mean and the standard deviation
    """
    inputs = [(seg[0], seg[1] - left_context, seg[2] + right_context,
               left_context, right_context) for seg in seg_list]
    MAX_WORKERS = 20
    pool = Pool(processes=MAX_WORKERS)
    if file_format == 'spro4':
        res = pool.map(segment_mean_std_spro4, sorted(inputs))
    elif file_format == 'htk':
        res = pool.map(segment_mean_std_htk, sorted(inputs))
    total_N = 0
    total_F = np.zeros(feature_size)
    total_S = np.zeros(feature_size)
    for N, F, S in res:
        total_N += N
        total_F += F
        total_S += S
    return total_N, total_F / total_N, total_S / total_N


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
        print(p_)
        p_.set_value(param_dict[p_.name])

def export_params(params, param_dict):
    """
    Export netork parameters into Numpy format

    :param params: dictionary of variables in Theano format
    :param param_dict: dictionary of variables in Numpy format
    """
    for k in param_dict:
        params[k.name] = k.get_value()

class FForwardNetwork(object):
    def __init__(self, filename=None,
                 input_size=0,
                 input_mean=np.empty(0),
                 input_std=np.empty(0),
                 hidden_layer_sizes=(),
                 layers_activations=(),
                 nclasses=0
                 ):
        if filename is not None:
            # Load DNN parameters
            #self.params = np.load(filename)
            self.params = dict()
            _p = np.load(filename)
            for k, v in _p.items():
                self.params[k] = v

            """ AJOUTER  DES VERIFICATIONS SUR LE CONTENU DU DICTIONNAIRE DE PARAMETRES"""

        else:  # initialize a NN with given sizes of layers and activation functions
            assert len(layers_activations) == len(hidden_layer_sizes) + 1, \
                "Mismatch between number of hidden layers and activation functions"

            sizes = (input_size,) + tuple(hidden_layer_sizes) + (nclasses,)

            self.params = {"input_mean": input_mean.astype(T.config.floatX),
                           "input_std": input_std.astype(T.config.floatX),
                           "activation_functions": layers_activations,
                           "b{}".format(len(sizes) - 1): np.zeros(sizes[-1]).astype(T.config.floatX),
                           "hidden_layer_sizes": hidden_layer_sizes
                           }

            for ii in range(1, len(sizes)):
                self.params["W{}".format(ii)] = np.random.randn(
                        sizes[ii - 1],
                        sizes[ii]).astype(T.config.floatX) * 0.1
                self.params["b{}".format(ii)] = np.random.random(sizes[ii]).astype(T.config.floatX) / 5.0 - 4.1
        
        init_logging()
        self.log = logging.getLogger()

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
            elif af == None:
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

    def train(self, training_seg_list,
              cross_validation_seg_list,
              feature_file_format,
              feature_size,
              feature_context=(7, 7),
              lr=0.008,
              segment_buffer_size=200,
              batch_size=512,
              max_iters=20,
              tolerance=0.003,
              output_file_name="",
              save_tmp_nnet=False):
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
        :param feature_file_format: spro4 or htk
        :param feature_size: dimension of the acoustic feature
        :param feature_context: tuple of left and right context given in
            number of frames
        :param lr: initial learning rate
        :param segment_buffer_size: number of segments loaded at once
        :param batch_size: size of the minibatches as number of frames
        :param max_iters: macimum number of epochs
        :param tolerance:
        :param output_file_name: root name of the files to save Neural Betwork parameters
        :param save_tmp_nnet: boolean, if True, save the parameters after each epoch
        """
        np.random.seed(42)

        # shuffle the training list
        shuffle_idx = np.random.permutation(np.arange(len(training_seg_list)))
        training_seg_list = [training_seg_list[idx] for idx in shuffle_idx]

        # If not done yet, compute mean and standard deviation on all training data
        if 0 in [len(self.params["input_mean"]), len(self.params["input_std"])]:
            import sys
            #if sys.version_info[0] >= 3:
            #if not os.path.exists("input_mean_std.npz"):
            if True:
                self.log.info("Compute mean and standard deviation from the training features")
                feature_nb, self.params["input_mean"], self.params["input_std"] = mean_std_many(feature_file_format,
                                                                                                feature_size,
                                                                                                training_seg_list,
                                                                                                feature_context[0],
                                                                                                feature_context[1])
                np.savez("input_mean_std", input_mean=self.params["input_mean"], input_std=self.params["input_std"])


            else:
                self.log.info("Load input mean and standard deviation from file")
                ms = np.load("input_mean_std.npz")
                self.params["input_mean"] = ms["input_mean"]
                self.params["input_std"] = ms["input_std"]


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
        last_cv_error = np.inf

        # Set the initial decay factor for the learning rate
        lr_decay_factor = 1

        # Iterate to train the network
        for kk in range(1, max_iters):
            lr *= lr_decay_factor  # update the learning rate

            error = accuracy = n = 0.0
            nfiles = 0

            # Iterate on the mini-batches
            for ii, training_segment_set in enumerate(training_segment_sets):
                l = []
                f = []
                for idx, val in enumerate(training_segment_set):
                    filename, s, e, label = val
                    e = s + len(label)
                    l.append(label)
                    f.append(sidekit.frontend.features.get_context(
                            sidekit.frontend.io.read_feature_segment(filename,
                                                                     feature_file_format,
                                                                     start=s - feature_context[0],
                                                                     stop=e + feature_context[1]),
                            left_ctx=feature_context[0],
                            right_ctx=feature_context[1],
                            apply_hamming=False))

                lab = np.hstack(l).astype(np.int16)
                fea = np.vstack(f).astype(np.float32)
                assert np.all(lab != -1) and len(lab) == len(fea)  # make sure that all frames have defined label
                shuffle = np.random.permutation(len(lab))
                lab = lab.take(shuffle, axis=0)
                fea = fea.take(shuffle, axis=0)

                nsplits = len(fea) / batch_size
                nfiles += len(training_segment_set)

                for jj, (X, t) in enumerate(zip(np.array_split(fea, nsplits), np.array_split(lab, nsplits))):
                    err, acc = train(X.astype(np.float32), t.astype(np.int16), lr)
                    error += err
                    accuracy += acc
                    n += len(X)
                self.log.info("%d/%d | %f | %f ", nfiles, len(training_seg_list), error / n, accuracy / n)

            error = accuracy = n = 0.0

            # Cross-validation
            for ii, cv_segment in enumerate(cross_validation_seg_list):
                filename, s, e, label = cv_segment
                e = s + len(label)
                t = label.astype(np.int16)
                X = sidekit.frontend.features.get_context(
                        sidekit.frontend.io.read_feature_segment(filename,
                                                                 feature_file_format,
                                                                 start=s - feature_context[0],
                                                                 stop=e + feature_context[1]),
                        left_ctx=feature_context[0],
                        right_ctx=feature_context[1],
                        apply_hamming=False)

                assert len(X) == len(t)
                err, acc = xentropy(X, t)
                error += err
                accuracy += acc
                n += len(X)

            # Save the current version of the network
            if save_tmp_nnet:
                tmp_dict = get_params(params_)
                tmp_dict.update({"activation_functions": self.params["activation_functions"]})
                np.savez(output_file_name + '_epoch' + str(kk), **tmp_dict)
                #np.savez(output_file_name + '_epoch' + str(kk), **get_params(params_))

            # Load previous weights if error increased
            if last_cv_error <= error:
                set_params(params_, last_params)
                error = last_cv_error

            # Start halving the learning rate or terminate the training
            if (last_cv_error - error) / np.abs([last_cv_error, error]).max() <= tolerance:
                if lr_decay_factor < 1:
                    break
                lr_decay_factor = 0.5

            # Update the cross-validation error
            last_cv_error = error

            # get last computed params
            last_params = get_params(params_)
            export_params(self.params, params_)

        # Save final network
        model_name = output_file_name + '_'.join([str(ii) for ii in self.params["hidden_layer_sizes"]])
        tmp_dict = get_params(params_)
        tmp_dict.update({"activation_functions": self.params["activation_functions"]})
        np.savez(output_file_name + '_epoch' + str(kk), **tmp_dict)
        #np.savez(model_name, **get_params(params_))

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
            elif af == None:
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

        # IL FAUT AJOUTER L'AFFICHAGE DE L'ARCHITECTURE DANS LE LOGGER
        return X_, Y_, params_

    def feed_forward(self, layer_number,
                     feature_file_list,
                     input_dir,
                     input_file_extension,
                     label_dir,
                     label_extension,
                     output_dir,
                     output_file_extension,
                     input_feature_format,
                     output_feature_format,
                     feature_context=(7, 7),
                     normalize_output="cmvn"):
        """
        Function used to extract bottleneck features or embeddings from an existing Neural Network.
        The first bottom layers of the neural network are loaded and all feature files are process through
        the network to get the output and save them as feature files.
        If specified, the output features can be normalized (cms, cmvn, stg) given input labels

        :param layer_number: number of layers to load from the model
        :param feature_file_list: list of feature files to process through the feed formward network
        :param input_dir: input directory where to load the features from
        :param input_file_extension: extension of the feature  files to load.
        :param label_dir: directory where to load the label files  from
        :param label_extension: extension of the label files to load
        :param output_dir: output directory where to save output features
        :param output_file_extension: extension of the output feature files
        :param input_feature_format: format of the feature files to read (htk or spro4)
        :param output_feature_format: format of the feature files to write (htk or spro4)
        :param feature_context: bi-dimensional tuple, context of the features to process, default is 7 features on the left and 7 on the right
        :param normalize_output: normalization applied to the output features, can be 'cms', 'cmvn', 'stg' or None
        """

        # Instantiate the network
        X_, Y_, params_ = self.instantiate_partial_network(layer_number)

        # Define the forward function to get the output of the first network: bottle-neck features
        forward = theano.function(inputs=[X_], outputs=Y_)

        # Iterate on the list of files, process the entire file and not only a segment
        start = None
        end = None
        if start is None:
            start = 0
        if end is None & feature_context[1] != 0:
            end = -2 * feature_context[1]

        # Create FeaturesServer to normalize the output features
        fs = sidekit.FeaturesServer(feat_norm=normalize_output)

        input_fn_model = input_dir + "{}" + input_file_extension
        lbl_fn_model = label_dir + "{}" + label_extension
        output_fn_model = output_dir + "{}" + output_file_extension
        for filename in feature_file_list:
            self.log.info("Process file %s", filename)
            bnf = forward(sidekit.frontend.features.get_context(
                    #sidekit.frontend.io.read_feature_segment(input_fn_model.format(filename),
                    #                                         input_feature_format,
                    #                                         start=start - feature_context[0],
                    #                                         stop=end + feature_context[1]),
                    sidekit.frontend.io.read_feature_segment(input_fn_model.format(filename),
                                                             input_feature_format,
                                                             start=start - feature_context[0],
                                                             stop = end + feature_context[1] if end is not None else None),
                    left_ctx=feature_context[0],
                    right_ctx=feature_context[1],
                    apply_hamming=False).astype(np.float32))

            # Load label file for feature normalization if needed
            speech_lbl = np.array([])
            if(os.path.exists(lbl_fn_model.format(filename))):
                speech_lbl = sidekit.frontend.read_label(lbl_fn_model.format(filename))

            # Normalize features using only speech frames
            if len(speech_lbl) == 0:
                self.log.warning("No label for %s", filename)
            else:
                if speech_lbl.shape[0] < bnf.shape[0]:
                    speech_lbl = np.hstack((speech_lbl, np.zeros(bnf.shape[0]-speech_lbl.shape[0], dtype='bool')))
                fs._normalize([speech_lbl], [bnf])

            # Save features in specified format
            if output_feature_format is "spro4":
                sidekit.frontend.write_spro4(bnf, output_fn_model.format(filename))
            elif output_feature_format is "htk":
                sidekit.frontend.write_htk(bnf, output_fn_model.format(filename))

    def display(self):
        """
        Display the structure of the feed-forward network in the standard output stream or in a logger object
        :param log: the logger object to feed
        """
        structure = "Network structure:\n\ninput size = {}\n   |\n   v\n".format(self.params["input_mean"].shape[0])
        for idx, l in enumerate(self.params["hidden_layer_sizes"]):
            structure += ("hidden layer {} size = {}\nActivation function: {}\n   |\n   v\n".format(idx, l,
                self.params["activation_functions"][idx]))
            structure += "output size = {}".format(
                    self.params["W{}".format(len(self.params["hidden_layer_sizes"]) -1)].shape[1])
            print(structure)

    def save(self):
        pass

    def compute_stat(self):
        pass

    def estimate_gmm(self):
        pass


"""
Tout ce qui suit est à convertir mais on vera plus tard
"""
# def compute_stat_dnn(nn_file_name, idmap, fb_dir, fb_extension='.fb',
#                 left_context=15, right_context=15, dct_nb=16, feature_dir='', 
#                 feature_extension='', viterbi=False):
#    """
#    :param nn_file_name: weights and biaises of the network stored in npz format
#    :param idmap: class name, session name and start/ stop information 
#        of each segment to process in an IdMap object
#      
#    :return: a StatServer...
#    """
#    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,floatX=float32'
#    # Load weight parameters and create a network
#    X_, Y_, params_ = create_theano_nn(np.load(nn_file_name))
#    # Define the forward function to get the output of the network
#    forward =  theano.function(inputs=[X_], outputs=Y_)
#
#    # Create the StatServer
#    ss = sidekit.StatServer(idmap)
#    
#
#    # Compute the statistics and store them in the StatServer
#    for idx, seg in enumerate(idmap.rightids):
#        # Load the features
#        traps = sidekit.frontend.features.get_trap(
#                    sidekit.frontend.io.read_spro4_segment(fb_dir + seg + fb_extension, 
#                                                       start=idmap.start[idx]-left_context, 
#                                                       end=idmap.stop[idx]+right_context), 
#                    left_ctx=left_context, right_ctx=right_context, dct_nb=dct_nb)
#
#        feat = traps
#        if feature_dir != '' or feature_extension != '':
#            feat = sidekit.frontend.io.read_spro4_segment(feature_dir + seg + feature_extension, 
#                                                       start=idmap.start[idx], 
#                                                       end=idmap.stop[idx])
#            if feat.shape[0] != traps.shape[0]:
#                raise Exception("Parallel feature flows have different length")
#
#        # Process the current segment and get the stat0 per frame
#        s0 = forward(traps)
#        if viterbi:
#            max_idx = s0.argmax(axis=1)            
#            z = np.zeros((s0.shape)).flatten()
#            z[np.ravel_multi_index(np.vstack((np.arange(30),max_idx)), s0.shape)] = 1.
#            s0 = z.reshape(s0.shape)
#   
#        sv_size = s0.shape[1] * feat.shape[1]
#        
#        # Store the statistics in the StatServer
#        if ss.stat0.shape == (0,):
#            ss.stat0 = np.empty((idmap.leftids.shape[0], s0.shape[1]))
#            ss.stat1 = np.empty((idmap.leftids.shape[0], sv_size))
#            
#        ss.stat0[idx, :] = s0.sum(axis=0)
#        ss.stat1[idx, :] = np.reshape(np.dot(feat.T, s0).T, sv_size)
#    
#    return ss
#        
#
# def compute_ubm_dnn(nn_weights, idmap, fb_dir, fb_extension='.fb',
#                 left_context=15, right_context=15, dct_nb=16, feature_dir='',
#                 feature_extension='', label_dir = '', label_extension='.lbl',
#                 viterbi=False):
#    """
#    """
#    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,floatX=float32'
#    # Accumulate statistics using the DNN (equivalent to E step)
#    
#    # Load weight parameters and create a network
#    #X_, Y_, params_ = create_theano_nn(np.load(nn_file_name))
#    X_, Y_, params_ = nn_weights
#    ndim =  params_[-1].get_value().shape[0]  # number of distributions
#    
#    print("Train a UBM with {} Gaussian distributions".format(ndim))    
#    
#    # Define the forward function to get the output of the network
#    forward =  theano.function(inputs=[X_], outputs=Y_)
#
#    # Create the StatServer
#    ss = sidekit.StatServer(idmap)
#    
#
#    # Initialize the accumulator given the size of the first feature file
#    if feature_dir != '' or feature_extension != '':
#        feat_dim = sidekit.frontend.io.read_spro4_segment(feature_dir + idmap.rightids[0] + feature_extension, 
#                                                       start=0, 
#                                                       end=2).shape[1]
#    else:
#        feat_dim = sidekit.frontend.features.get_trap(
#                    sidekit.frontend.io.read_spro4_segment(fb_dir + idmap.rightids[0] + fb_extension, 
#                                                       start=0, 
#                                                       end=2), 
#                    left_ctx=left_context, right_ctx=right_context, dct_nb=dct_nb).shape[1]
#    
#    # Initialize one Mixture for UBM storage and one Mixture to accumulate the 
#    # statistics
#    ubm = sidekit.Mixture()
#    ubm.cov_var_ctl = np.ones((ndim, feat_dim))
#    
#    accum = sidekit.Mixture()
#    accum.mu = np.zeros((ndim, feat_dim))
#    accum.invcov = np.zeros((ndim, feat_dim))
#    accum.w = np.zeros(ndim)
#
#    # Compute the zero, first and second order statistics
#    for idx, seg in enumerate(idmap.rightids):
#        
#        start = idmap.start[idx]
#        end = idmap.stop[idx]
#        if start is None:
#            start = 0
#        if end is None:
#            endFeat = None
#            end = -2 * right_context
#        
#        
#        # Load speech labels
#        speech_lbl = sidekit.frontend.read_label(label_dir + seg + label_extension)
#        
#        # Load the features
#        traps = sidekit.frontend.features.get_trap(
#                    sidekit.frontend.io.read_spro4_segment(fb_dir + seg + fb_extension, 
#                                                       start=start-left_context, 
#                                                       end=end+right_context), 
#                    left_ctx=left_context, right_ctx=right_context, dct_nb=dct_nb)[speech_lbl, :]
#
#        feat = traps
#        if feature_dir != '' or feature_extension != '':
#            feat = sidekit.frontend.io.read_spro4_segment(feature_dir + seg + feature_extension, 
#                                                       start=max(start, 0), 
#                                                       end=endFeat)[speech_lbl, :]
#            if feat.shape[0] != traps.shape[0]:
#                raise Exception("Parallel feature flows have different length")
#
#        # Process the current segment and get the stat0 per frame
#        s0 = forward(traps)
#        if viterbi:
#            max_idx = s0.argmax(axis=1)            
#            z = np.zeros((s0.shape)).flatten()
#            z[np.ravel_multi_index(np.vstack((np.arange(30),max_idx)), s0.shape)] = 1.
#            s0 = z.reshape(s0.shape)
#   
#        sv_size = s0.shape[1] * feat.shape[1]
#        
#        # zero order statistics
#        accum.w += s0.sum(0)
#
#        #first order statistics
#        accum.mu += np.dot(feat.T, s0).T
#
#        # second order statistics
#        accum.invcov += np.dot(np.square(feat.T), s0).T     
#
#    # M step    
#    ubm._maximization(accum)
#    
#    return ubm
