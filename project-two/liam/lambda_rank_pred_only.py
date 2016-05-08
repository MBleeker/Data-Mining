import csv
import cPickle
import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from itertools import count
from query import *
from ndcg import *

MOMENTUM = 0.95

class LambdaRankPred:

    NUM_INSTANCES = count()

    def __init__(self, feature_count, width=[40,20,5]):
        self.x_mean_train, self.x_std_train, \
            self.y_mean_train, self.y_std_train = 1., 1., 1., 1.
        self.normalise = True
        self.width = width
        self.feature_count = feature_count
        self.output_layer = self.build_model(feature_count,1)
        self.iter_funcs = self.create_functions(self.output_layer)

    def predict(self, feature_vectors):
        if self.normalise:
            feature_vectors = self.normalise_X(feature_vectors)
        scores = self.iter_funcs['out'](feature_vectors)
        return scores

    def build_model(self, input_dim, output_dim):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print "input_dim:",input_dim, "output_dim:",output_dim
        network = lasagne.layers.InputLayer(
            shape=(1, input_dim)
        )
        # Hidden layers and dropout:
        hidden_nonlin = lasagne.nonlinearities.leaky_rectify
        output_nonlin=lasagne.nonlinearities.linear
        for layer in range(len(self.width)):
            network = lasagne.layers.DenseLayer(
                    network, self.width[layer], nonlinearity=hidden_nonlin)
        # Output layer:
        network = lasagne.layers.DenseLayer(network, 
                                    1, nonlinearity=output_nonlin)
        return network
        
    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer, pointwise = False,
                          X_tensor_type=T.matrix, momentum=MOMENTUM,
                          L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """
        
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, 
                                                   X_batch, deterministic=True,
                                                   dtype="float32")

        # I've somehow managed to 'invert' the pairwise models, such that higher
        # NN outputs (scores) = lower relevance. Rather than going into the details
        # and trying to fix this, I've just flipped the pointwise loss so that
        # it behaves the same - notice the '-' in front of output here:
        loss_train = lasagne.objectives.squared_error(output, y_batch)

        loss_train = loss_train.mean()

        # TODO: (Optionally) You can add regularization if you want - for those interested
        # L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        # L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        # loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        all_params = lasagne.layers.get_all_params(output_layer)
        updates = lasagne.updates.adam(loss_train, all_params,
                                       learning_rate = 0.01)
        score_func = theano.function(
            [X_batch],output_row_det,allow_input_downcast=True
        )
        train_func = theano.function(
            [X_batch,y_batch], loss_train,
            updates=updates,allow_input_downcast=True
        )

        return dict(
            train=train_func,
            out=score_func,
        )
    
    def get_norm_constants(self):
        return self.x_mean_train, self.x_std_train, \
               self.y_mean_train, self.y_std_train
    
    def set_norm_constants(self, xm, xs, ym, ys):
        self.x_mean_train, self.x_std_train, \
            self.y_mean_train, self.y_std_train = xm, xs, ym, ys
            
    def calc_norm_constants(self):
        stack = np.vstack(self.train_queries.get_feature_vectors())
        self.x_mean_train = np.mean(stack, axis = 0)
        self.x_std_train = np.std(stack, axis = 0)
            
    def normalise_X(self, X, reset = False):
        X = X - self.x_mean_train
        X = X / self.x_std_train
        return X
        
    def normalise_y(self, y, reset = False):
        y = y - self.y_mean_train
        y = y / self.y_std_train
        return y
        
    def denormalise_y(self, preds):
        return self.y_mean_train + self.y_std_train * preds