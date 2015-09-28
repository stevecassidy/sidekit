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
Copyright 2014-2015 Anthony Larcher

:mod:`theano_utils` provides utilities to facilitate the work with SIDEKIT
and THEANO.
"""

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2015-2016 Anthony Larcher"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'

import numpy as np
import scipy as sp
import pickle
import gzip
import os

import random
import sidekit
import sys
import logging
import errno
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,floatX=float32'
import theano, theano.tensor as T

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def create_theano_nn(param_dict):
    """
    """
    X_ = T.matrix("X")
    mean_ = theano.shared(param_dict['input_mean'], name='input_mean')
    std_  = theano.shared(param_dict['input_std'], name='input_std')
    Y_ = (X_ - mean_) / std_
    params_ = [mean_, std_]
    n_hidden_layers = len(param_dict.keys())/2-2
    for ii, f in enumerate([T.nnet.sigmoid]*n_hidden_layers+[T.nnet.softmax]):
        W_ = theano.shared(param_dict['W'+str(ii+1)], name='W'+str(ii+1))
        b_ = theano.shared(param_dict['b'+str(ii+1)], name='b'+str(ii+1))
        Y_ = f(Y_.dot(W_) + b_)
        params_ += [W_, b_] 
    return X_, Y_, params_


def init_params(input_mean, input_std, hidden_layer_sizes, nclasses):
    """
    """
    sizes = (len(input_mean),)+tuple(hidden_layer_sizes)+(nclasses,)
    params_dict = {"input_mean": input_mean.astype(T.config.floatX), "input_std": input_std.astype(T.config.floatX)}
    for ii in range(1,len(sizes)):   params_dict['W'+str(ii)] = np.random.randn(sizes[ii-1],sizes[ii]).astype(T.config.floatX)*0.1
    for ii in range(1,len(sizes)-1): params_dict['b'+str(ii)] = np.random.random(           sizes[ii]).astype(T.config.floatX)/5.0-4.1
    params_dict['b'+str(len(sizes)-1)] = np.zeros(sizes[len(sizes)-1]).astype(T.config.floatX)
    return params_dict


def get_params(params_):
    return {p.name: p.get_value() for p in params_}


def set_params(params_, param_dict):
    for p_ in params_: p_.set_value(param_dict[p_.name])


def compute_stat_dnn(nn_file_name, idmap, fb_dir, fb_extension='.fb',
                 left_context=15, right_context=15, dct_nb=16, feature_dir='', 
                 feature_extension='', viterbi=False):
    """
    :param nn_file_name: weights and biaises of the network stored in npz format
    :param idmap: class name, session name and start/ stop information 
        of each segment to process in an IdMap object
      
    :return: a StatServer...
    """       
    # Load weight parameters and create a network
    X_, Y_, params_ = create_theano_nn(np.load(nn_file_name))
    # Define the forward function to get the output of the network
    forward =  theano.function(inputs=[X_], outputs=Y_)

    # Create the StatServer
    ss = sidekit.StatServer(idmap)
    

    # Compute the statistics and store them in the StatServer
    for idx, seg in enumerate(idmap.rightids):
        # Load the features
        traps = sidekit.frontend.features.get_trap(
                    sidekit.frontend.io.read_spro4_segment(fb_dir + seg + fb_extension, 
                                                       start=idmap.start[idx]-left_context, 
                                                       end=idmap.stop[idx]+right_context), 
                    left_ctx=left_context, right_ctx=right_context, dct_nb=dct_nb)

        feat = traps
        if feature_dir != '' or feature_extension != '':
            feat = sidekit.frontend.io.read_spro4_segment(feature_dir + seg + feature_extension, 
                                                       start=idmap.start[idx], 
                                                       end=idmap.stop[idx])
            if feat.shape[0] != traps.shape[0]:
                raise Exception("Parallel feature flows have different length")

        # Process the current segment and get the stat0 per frame
        s0 = forward(traps)
        if viterbi:
            max_idx = s0.argmax(axis=1)            
            z = np.zeros((s0.shape)).flatten()
            z[np.ravel_multi_index(np.vstack((np.arange(30),max_idx)), s0.shape)] = 1.
            s0 = z.reshape(s0.shape)
   
        sv_size = s0.shape[1] * feat.shape[1]
        
        # Store the statistics in the StatServer
        if ss.stat0.shape == (0,):
            ss.stat0 = np.empty((idmap.leftids.shape[0], s0.shape[1]))
            ss.stat1 = np.empty((idmap.leftids.shape[0], sv_size))
            
        ss.stat0[idx, :] = s0.sum(axis=0)
        ss.stat1[idx, :] = np.reshape(np.dot(feat.T, s0).T, sv_size)
    
    return ss
        

def compute_ubm_dnn(nn_weights, idmap, fb_dir, fb_extension='.fb',
                 left_context=15, right_context=15, dct_nb=16, feature_dir='',
                 feature_extension='', label_dir = '', label_extension='.lbl',
                 viterbi=False):
    """
    """     
    # Accumulate statistics using the DNN (equivalent to E step)
    
    # Load weight parameters and create a network
    #X_, Y_, params_ = create_theano_nn(np.load(nn_file_name))
    X_, Y_, params_ = nn_weights
    ndim =  params_[-1].get_value().shape[0]  # number of distributions
    
    print("Train a UBM with {} Gaussian distributions".format(ndim))    
    
    # Define the forward function to get the output of the network
    forward =  theano.function(inputs=[X_], outputs=Y_)

    # Create the StatServer
    ss = sidekit.StatServer(idmap)
    

    # Initialize the accumulator given the size of the first feature file
    if feature_dir != '' or feature_extension != '':
        feat_dim = sidekit.frontend.io.read_spro4_segment(feature_dir + idmap.rightids[0] + feature_extension, 
                                                       start=idmap.start[0], 
                                                       end=idmap.stop[0]).shape[1]
    else:
        feat_dim = sidekit.frontend.features.get_trap(
                    sidekit.frontend.io.read_spro4_segment(fb_dir + idmap.rightids[0] + fb_extension, 
                                                       start=idmap.start[0], 
                                                       end=idmap.stop[0]), 
                    left_ctx=left_context, right_ctx=right_context, dct_nb=dct_nb).shape[1]
    
    # Initialize one Mixture for UBM storage and one Mixture to accumulate the 
    # statistics
    ubm = sidekit.Mixture()
    ubm.cov_var_ctl = np.ones((ndim, feat_dim))
    
    accum = sidekit.Mixture()
    accum.mu = np.zeros((ndim, feat_dim))
    accum.invcov = np.zeros((ndim, feat_dim))
    accum.w = np.zeros(ndim)

    # Compute the zero, first and second order statistics
    for idx, seg in enumerate(idmap.rightids):
        
        start = idmap.start[idx]
        end = idmap.stop[idx]
        if start is None:
            start = 0
        if end is None:
            end = -2 * right_context
        
        
        # Load speech labels
        speeh_lbl = sidekit.frontend.read_label(label_dir + seg + label_extension)
        
        # Load the features
        traps = sidekit.frontend.features.get_trap(
                    sidekit.frontend.io.read_spro4_segment(fb_dir + seg + fb_extension, 
                                                       start=start-left_context, 
                                                       end=end+right_context), 
                    left_ctx=left_context, right_ctx=right_context, dct_nb=dct_nb)[speech_lbl, :]

        feat = traps
        if feature_dir != '' or feature_extension != '':
            feat = sidekit.frontend.io.read_spro4_segment(feature_dir + seg + feature_extension, 
                                                       start=idmap.start[idx], 
                                                       end=idmap.stop[idx])[speech_lbl, :]
            if feat.shape[0] != traps.shape[0]:
                raise Exception("Parallel feature flows have different length")

        # Process the current segment and get the stat0 per frame
        s0 = forward(traps)
        if viterbi:
            max_idx = s0.argmax(axis=1)            
            z = np.zeros((s0.shape)).flatten()
            z[np.ravel_multi_index(np.vstack((np.arange(30),max_idx)), s0.shape)] = 1.
            s0 = z.reshape(s0.shape)
   
        sv_size = s0.shape[1] * feat.shape[1]
        
        # zero order statistics
        accum.w += s0.sum(0)

        #first order statistics
        accum.mu += np.dot(feat.T, s0).T

        # second order statistics
        accum.invcov += np.dot(np.square(feat.T), s0).T     

    # M step    
    ubm._maximization(accum)
    
    return ubm