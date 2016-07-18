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
import logging
import numpy
import os
import timeit
from sidekit.sidekit_io import init_logging
import theano
import theano.tensor as T

# Warning, FUEL is needed in this version, we'll try to remove this dependency in the future
import fuel
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, ConstantScheme
from fuel.transformers import Mapping, Batch, Padding, Filter, Unpack, AddContext, StackAndShuffle, Cache, ScaleAndShift
from fuel.streams import DataStream

log = logging.getLogger()


#######################################################################################################################
# DEFINE DISTANCES TO USE IN THE TRIPLET RANKING LAYER
#######################################################################################################################
def add(x,y):
    return x+y


def _squared_magnitude(x):
    return T.sqr(x).sum(axis=-1)


def _magnitude(x):
    return T.sqrt(T.maximum(_squared_magnitude(x), numpy.finfo(x.dtype).tiny))


def cosine_similarity( x, y):
    return (x * y).sum(axis=-1) / (_magnitude(x) * _magnitude(y))


def euclidean(x, y):
    return _magnitude(x - y)


def squared_euclidean(x, y):
    return _squared_magnitude(x - y)


def dot_prod(x,y):
    return (x * y).sum(axis=-1)


#######################################################################################################################
# DEFINE A CLASS TO MANAGE A SIMPLE HIDDEN LAYER
#######################################################################################################################
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.input = input
        #
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in Glorot & Bengio (2010)
        #        suggest that you should use 4 times larger initial weights
        #        for sigmoid compared to tanh
        #W_values=None

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values = numpy.asarray(rng.normal(size=(n_in, n_out)) * 0.1, dtype=theano.config.floatX)

            # ajout pour debug
            #W_values = numpy.hstack([numpy.eye(n_in).astype(T.config.floatX), numpy.zeros((n_in, n_out-n_in), dtype=T.config.floatX)])
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            #b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b_values = (numpy.random.random((n_out,)) / 5.0 - 4.1).astype(T.config.floatX) 
            # ajout pour debug
            #b_values = (numpy.zeros(n_out)).astype(T.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

#######################################################################################################################
# DEFINE THE TRIPLET RANKING LAYER
#######################################################################################################################
class TripletRankingLossLayer(object):
    """Triplet ranking loss layer Class
    """

    def __init__(self, input_example, input_positive, input_negative, margin):
        """ Initialize the parameters of the triplet ranking loss layer

        :type input: theano.tensor.TensorType
        :param input_example: the original example to classify

        :type input: theano.tensor.TensorType
        :type input_positive : associated positive example

        :type input: theano.tensor.TensorType
        :param input_embeddingsLWrong: Letter n-gram embeddings (one minibatch), correponds to the wrong words
        """
        # keep track of model input and target.
        # We store a flattened (vector) version of target as y, which is easier to handle
        self.input_example = input_example
        self.input_positive = input_positive
        self.input_negative = input_negative
        #self.input_LossRank = input_LossRank   # IL S'AGIT D'UN POIDS DONNeE PAR TRIPLET LORS DU CALCUL DU COUT (on peut mettre 1.0 dans un premier temps)
        self.margin = margin

        self.positive_distances = dot_prod(self.input_example, self.input_positive)
        self.negative_distances = dot_prod(self.input_example, self.input_negative)


    def RankingLoss(self):
        """Return the mean of the triplet ranking loss"""
        #loss = T.mean(self.input_LossRank * (T.maximum(0, self.margin - self.positive_distances + self.negative_distances)))
        loss = T.mean(T.maximum(0, self.margin - self.positive_distances + self.negative_distances))
        return loss

    def distantce(self):
        l= T.maximum(0, self.margin - self.positive_distances + self.negative_distances)
        return [self.positive_distances,self.negative_distances,l]


###############################################################################################################################
#  DEFINE THE TRIPLET ARCHITECTURE
###############################################################################################################################
class TRIPLE_MLP(object):
    """Triple Multi-Layer Perceptron Class with a triplet ranking loss layer on top
    A single architecture is defined, three parallel networkds are defined and share the same parameters
    """
    def __init__(self, rng, input_example, input_positive, input_negative, n_in, n_hidden, activation_functions):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights and bias

        :type input: theano.tensor.TensorType
        :param input_example: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type input: theano.tensor.TensorType
        :type input_positive: symbolic variable that describes the positive input (one minibatch)

        :type input: theano.tensor.TensorType
        :type input_negative: symbolic variable that describes the negative input (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        1which the datapoints lie

        :type n_hidden: list of int
        :param n_hidden: number of hidden units in each hidden layer

        :type activation_functions: list of theano.Op or function
        :param activation_functions: list of non linearity to be applied in each hidden layers
        """
        # keep track of model input and loss rank function
        self.input_example=input_example
        self.input_positive=input_positive
        self.input_negative=input_negative

        # Build all necessary hidden layers and chain them
        # For each layer, a first instance is created and initialize,
        # the two parallel layers are then build and share the same weight and bias
        self.hidden_layers = []
        layer_n_in = n_in
        layer_input = [input_example, input_positive, input_negative]

        self.params = []
        self.weights = []
        self.bias = []

        for ii, activation in enumerate(activation_functions):

            # instantiate the first column of the architecture
            hidden_layer_E = HiddenLayer(
                    rng=rng,
                    input=layer_input[0],
                    n_in=layer_n_in,
                    n_out=n_hidden[ii],
                    activation=activation)
            print("taille de l entree: {}".format(layer_input[0].shape))
            print("layer  {}, input = {}, output = {}".format(ii, layer_n_in, n_hidden[ii]))

            self.params += hidden_layer_E.params

            # Instantiate the positive column using the sale weight and bias
            hidden_layer_P = HiddenLayer(
                    rng=rng,
                    input=layer_input[1],
                    n_in=layer_n_in,
                    n_out=n_hidden[ii],
                    activation=activation,
                    W=hidden_layer_E.W,
                    b=hidden_layer_E.b)

            # instantiate the negative column using the same weight and biais
            hidden_layer_N = HiddenLayer(
                    rng=rng,
                    input=layer_input[2],
                    n_in=layer_n_in,
                    n_out=n_hidden[ii],
                    activation=activation,
                    W=hidden_layer_E.W,
                    b=hidden_layer_E.b)

            self.hidden_layers.append(hidden_layer_E)
            self.hidden_layers.append(hidden_layer_P)
            self.hidden_layers.append(hidden_layer_N)

            # Prepare data for the next layer
            layer_input[0] = hidden_layer_E.output
            layer_input[1] = hidden_layer_P.output
            layer_input[2] = hidden_layer_N.output
            layer_n_in = n_hidden[ii]

        # The Triplet Loss layer gets as input the hidden units of the hidden layer,
        # from the three columns
        print("nb layers: {}".format(len(self.hidden_layers)//3))
        self.Triplet_Rank_Layer = TripletRankingLossLayer(
            input_example=layer_input[0],
            input_positive=layer_input[1],
            input_negative=layer_input[2],
            margin=1)

        # self.params has all the parameters of the model,
        # self.weights contains only the `W` variables.
        # We also give unique name to the parameters, this will be useful to save them.

        layer_idx = 0
        for idx, hl in enumerate(self.hidden_layers[::3]):
             hl.W.name = 'W_{}'.format(idx)
             hl.b.name = 'b_{}'.format(idx)
             self.weights.append(hl.W)
             self.bias.append(hl.b)

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = sum(abs(W).sum() for W in self.weights)

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = sum((W ** 2).sum() for W in self.weights)

    def Triplet_Rank_loss(self):
        return self.Triplet_Rank_Layer.RankingLoss()

    def errors(self):
        # same holds for the function computing the number of errors
        return self.Triplet_Rank_Layer.RankingLoss()

    def distances(self):
        # function returns the distances (w+,e)  and (w-,e)
        return self.Triplet_Rank_Layer.distantce()


#######################################################################################################################
# DEFINE FUNCTIONS REQUIRED FOR TRAINING
#######################################################################################################################
def relu(x):
    return x * (x > 0)


def regularized_cost_grad(mlp_model, L1_reg, L2_reg):
    loss = (mlp_model.Triplet_Rank_loss() +
            L1_reg * mlp_model.L1 +
            L2_reg * mlp_model.L2_sqr)
    params = mlp_model.params
    grads = theano.grad(loss, wrt=params)
    # Return (param, grad) pairs
    return zip(params, grads)


def get_momentum_updates(params_and_grads, lr, rho):
    res = []
    # numpy will promote (1 - rho) to float64 otherwise
    one = numpy.float32(1.)
    zero = numpy.float32(0.)

    for p, g in params_and_grads:
        #up = theano.shared(p.get_value() * 0)
        up = theano.shared(p.get_value() * zero)
        res.append((p, p - lr * up))
        res.append((up, rho * up + (one - rho) * g))

    return res


def get_momentum_training_fn(triplet_model, L1_reg, L2_reg, lr, rho):
    inputs = [triplet_model.input_example, triplet_model.input_positive, triplet_model.input_negative]
    params_and_grads = regularized_cost_grad(triplet_model, L1_reg, L2_reg)
    updates = get_momentum_updates(params_and_grads, lr=lr, rho=rho)
    return theano.function(inputs, updates=updates)


def get_test_fn(triplet_model):
    return theano.function([triplet_model.input_example,triplet_model.input_positive, triplet_model.input_negative],
                           triplet_model.errors())


def get_distances_fn(triplet_model):
    return theano.function([triplet_model.input_example, triplet_model.input_positive, triplet_model.input_negative],
                           triplet_model.distances(),on_unused_input='ignore')


#######################################################################################################################
# TRAINING FUNCTION
#######################################################################################################################

# distance_fn : fonction qui calcule la distance
# train_model : fonction d'apprentissage
# test_model : fonction de test
# train_set :
# Vocab_set :

def triplet_training(triplet_model, distance_fn,
                     train_model, test_model, train_set, validation_set,
                     model_name='triplet_model',
                     # maximum number of epochs
                     n_epochs=1000,
                     # look at this many examples regardless
                     patience=2000,  # initialement à 200
                     # wait this much longer when a new best is found
                     patience_increase=10,
                     # a relative improvement of this much is considered significant
                     improvement_threshold=0.995,
                     batch_size=50):

    #filename = 'Letter_3gram_500DEmb_acoustiqEmbed_score38.495_LossRank_C.hdf5'#Letter_3gram_500DEmb_acoustiqEmbed_score44.451_LossRang.hdf5'
    n_train_batches = train_set.num_examples // batch_size

    # on utilise un data_stream de FUEL

    #CREER LE DATA_STREAM A PARTIR DU train_set
    train_stream = DataStream.default_stream(train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size))
    validation_stream = DataStream.default_stream(validation_set, iteration_scheme=ShuffledScheme(validation_set.num_examples, batch_size))
    #train_stream = Cast (Flatten(DataStream.default_stream(train_set,
    #                                                       iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size))),
    #                     dtype='float32', which_sources=('features','embeddings'))

    # go through this many minibatches before checking the network on the validation set;
    # in this case we check every epoch
    #print("num exemple train = {}".format(train_set.num_examples))
    log = logging.getLogger()
    log.info("batch size = %d", batch_size)
    log.info("ntrain_batch = %d", n_train_batches)
    #print("batch size = {}".format(batch_size))
    #print("ntrain batch  = {}".format(n_train_batches))
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()
    validation_losses=[]
    distances=[]
    done_looping = False
    epoch = 0
    #j=0
    best_distances_postives=1.0
    #print("n_epochs = {}".format(n_epochs))
    log.info("n_epochs = %d", n_epochs)
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        minibatch_index = 0

        validation_losses=[]
        distancespositives=[]
        loss_w=[]
        distances=0

        distancesnegatives=[]
        """
        train_stream doit renvoyer des donnees qui permettent de generer des triplets (example, positif, negatif)
        Pour l instant on n utilise pas de loss_rank
        """
        for minibatch_example,  minibatch_positive, minibatch_negative in train_stream.get_epoch_iterator():

 
            #print("example: {}, positive: {}, negative: {}".format(minibatch_example.shape, minibatch_positive.shape, minibatch_negative.shape))
            #target = numpy.empty(minibatch_example.shape[0])
            #nontarget = numpy.empty(minibatch_example.shape[0])
            #loss = numpy.empty(minibatch_example.shape[0])
            #for k in range(minibatch_example.shape[0]):
            #    target[k] = 1 - numpy.dot(minibatch_example[k, :], minibatch_positive[k, :])
            #    nontarget[k] = 1 - numpy.dot(minibatch_example[k, :], minibatch_negative[k, :])
            #    loss[k] = 1 - target[k] + nontarget[k]
            #print("minibatch dot product; target = {}, nontarget = {}, loss = {}".format(target.mean(), nontarget.mean(), loss.mean()))

            train_model(minibatch_example, minibatch_positive, minibatch_negative)

            if (minibatch_index % 10) == 0:
                v_losses = test_model(minibatch_example, minibatch_positive, minibatch_negative)
                print("minibatch {}, Validation loss = {}".format(minibatch_index, v_losses))

                distances= distance_fn(minibatch_example, minibatch_positive, minibatch_negative)
                print("minibatch {}, distance positive = {}, distance negative = {}, l = {}".format(minibatch_index, distances[0].mean(), distances[1].mean(), distances[2].mean()))
                #target = numpy.empty(minibatch_example.shape[0])
                #nontarget = numpy.empty(minibatch_example.shape[0])
                #loss = numpy.empty(minibatch_example.shape[0])
                #for k in range(minibatch_example.shape[0]):
                #    target[k] = numpy.dot(minibatch_example[k, :], minibatch_positive[k, :])
                #    nontarget[k] = numpy.dot(minibatch_example[k, :], minibatch_negative[k, :])
                #    loss[k] = target[k] + nontarget[k]
                #print("minibatch {}, target = {}, nontarget = {}, loss = {}".format(minibatch_index,target.mean(), nontarget.mean(), loss.mean()))

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set

                for minibatch_example, minibatch_positive, minibatch_negative in validation_stream.get_epoch_iterator():

                    # loss_rank=loss_rank.reshape(1,batch_size)
                    loss_rank = numpy.ones((batch_size,), dtype = 'float32')

                    validation_losses.append(test_model(minibatch_example, minibatch_positive, minibatch_negative))

                    distances= distance_fn(minibatch_example, minibatch_positive, minibatch_negative)

                    distancespositives.append(numpy.sum(distances[0]))
                    # distancesnegatives.append(numpy.sum(distances[1] * loss_rank)/numpy.sum(loss_rank))
                    loss_w.append(numpy.sum(distances[2]))

                this_validation_loss = numpy.mean(validation_losses)

                validation_losses=[]
                log.info("epoch %d, minibatch %d / %d, train loss %f", epoch, minibatch_index + 1, n_train_batches, this_validation_loss)
                log.info("%f, Weighted loss, %f", numpy.mean(distancespositives), numpy.mean(loss_w))

                log.info("epoch {}, train error = {}, positive distance = {}, Weighted loss = {}".format(epoch,
                                                                                 this_validation_loss,
                                                                                 numpy.mean(distancespositives),
                                                                                 numpy.mean(loss_w)))

                # if we got the best validation score until now
                if  this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_distances_postives = numpy.mean(distancespositives)

                        best = {param.name: param.get_value() for param in triplet_model.params}
                        numpy.savez('param/best_{}_bestdevscore.npz'.format(model_name, **best))

            minibatch_index += 1
            #print " patience =",patience
            #print " iter = ", iter
            #if(epoch % 10 == 0):
            #    savParamFileName = 'param/Current_H2_2_2_WARPprShuff_dot_C_mlp_momentum_param'+filename +'train-score'+str(this_validation_loss)+'epoch_'+str(epoch)+'bs_'+str(batch_size)+'_lr_'+str(lr)+'postdist'+str(numpy.mean(distancespositives))+'_H1-300-H2-100_marg1_Bfor.pkl'
            #    save_param(savParamFileName)
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    log.info('Optimization complete with best train_score score of train score {} %, '.format(best_validation_loss))
    #print('Optimization complete with best train_score score of train score {} %, '.format(best_validation_loss))


    log.info('The code ran for {} epochs, with {} epochs/sec ({:02} total time)'.format(epoch,
                                                                                     1. * epoch / (end_time - start_time),
                                                                                     (end_time - start_time) / 60.))
