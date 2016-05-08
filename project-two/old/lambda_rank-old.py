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

class LambdaRank:

    NUM_INSTANCES = count()

    def __init__(self, feature_count, mod_type = 'Pointwise', learn = 0.001,
                 batch_size = 768, train_queries = None):
        self.lr = learn
        self.sigma = 1.
        self.model_type = mod_type
        self.feature_count = feature_count
        self.batch_size = batch_size
        self.output_layer = self.build_model(feature_count,1)
        self.iter_funcs = self.create_functions(self.output_layer)
        self.uv_pairs = None
        self.norming_dcgs = None
        self.train_queries = train_queries
        if self.train_queries != None:
            self.init_uv_pairs()
        
    def init_uv_pairs(self):
        self.uv_pairs = {}
        self.s_uv_sets = {}
        self.norming_dcgs = {}
        for q in self.train_queries.get_qids():
            labels = self.train_queries[q].get_labels()
            self.norming_dcgs[q] = dcg_at_k(sorted(labels, reverse=True), 
                                       len(labels))
            uv_pairs_u_greater_v = []
            labels = self.train_queries.get_query(q).get_labels()
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if labels[i] > labels[j]:
                        uv_pairs_u_greater_v.append([i, j])
            self.uv_pairs[q] = uv_pairs_u_greater_v
            self.s_uv_sets[q] = self.get_suvs(labels)      

    def get_suvs(self, labels):
        s_uv = np.zeros((len(labels),len(labels)))
        for u in range(len(labels)):
            for v in range(len(labels)):
                if labels[u] > labels[v]:
                    s_uv[u, v] = 1
                elif labels[u] < labels[v]:
                    s_uv[u, v] = -1
                else:
                    s_uv[u, v] = 0
        return s_uv

    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries = None, num_epochs = 1):
        if train_queries != None:
            self.train_queries = train_queries
            self.init_uv_pairs()
        elif self.train_queries == None:
            print "You haven't provided any train queries yet!"
            raise
        try:
            now = time.time()
            for epoch in self.train():
                if epoch['number'] % 5 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                    now = time.time()
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores

    def score_matrix(self, feature_vectors):
        scores = self.iter_funcs['out'](feature_vectors)
        return scores

    def build_model(self,input_dim, output_dim):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print "input_dim:",input_dim, "output_dim:",output_dim
        l_in = lasagne.layers.InputLayer(
            shape=(self.batch_size, input_dim),
        )

        l_hid = lasagne.layers.DenseLayer(
            l_in,
            num_units=40,
            nonlinearity=lasagne.nonlinearities.rectify,
        )

        l_hid = lasagne.layers.DenseLayer(
            l_hid,
            num_units=20,
            nonlinearity=lasagne.nonlinearities.rectify,
        )

        l_hid = lasagne.layers.DenseLayer(
            l_hid,
            num_units=5,
            nonlinearity=lasagne.nonlinearities.rectify,
        )

        l_out = lasagne.layers.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out
    def save_params(self):
        pass
    def load_params(self, fname):
        
        pass
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

        output_row_det = lasagne.layers.get_output(output_layer, X_batch,deterministic=True, dtype="float32")

        # I've somehow managed to 'invert' the pairwise models, such that higher
        # NN outputs (scores) = lower relevance. Rather than going into the details
        # and trying to fix this, I've just flipped the pointwise loss so that
        # it behaves the same - notice the '-' in front of output here:
        if self.model_type == 'Pointwise':        
            loss_train = lasagne.objectives.squared_error(output, y_batch)
        else:
            loss_train = (output * y_batch)

        loss_train = loss_train.mean()

        # TODO: (Optionally) You can add regularization if you want - for those interested
        # L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        # L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        # loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        all_params = lasagne.layers.get_all_params(output_layer)
        updates = lasagne.updates.adam(loss_train, all_params,
                                       learning_rate = self.lr)
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

    def get_dcg_with_labels_switched(self, labels, u, v):
        a = 2 ** (labels[v] - 1)
        b = 2 ** (labels[u] - 1)
        return (a - b) / np.log2(2 + u) + (b - a) / np.log2(2 + v)
        
    def get_single_lambda_uv(self, u, v, s_uvs, scores):
        return self.sigma * 1. * (0.5 * (1. - s_uvs[u, v]) - \
                1. / (1. + np.exp(self.sigma * (scores[u] - scores[v]))))
        
    def get_single_lambda_rank_uv(self, u, v, s_uvs, scores):
        # Equation 6 in Burges paper
        return self.sigma * 1. / (1. + \
                                  np.exp(self.sigma * (scores[u] - scores[v])))
        
    def compute_lambdas(self, labels, scores, uv_pairs_u_greater_v, s_uvs, 
                        normalising_dcg):
        lambdas_out = np.zeros(len(labels))
        if self.model_type == 'LambdaRank':
            if normalising_dcg != 0.:
                for u, v in uv_pairs_u_greater_v:
                    ndcg_chg = self.get_dcg_with_labels_switched(labels, u, v)
                    ndcg_chg = np.abs(ndcg_chg / normalising_dcg)
                    #lambda_uv = self.get_single_lambda_rank_uv(u, v, s_uvs, scores) * ndcg_chg * 100.                    
                    lambda_uv = self.get_single_lambda_uv(u, v, s_uvs, scores) * ndcg_chg * 100.
                    lambdas_out[u] += lambda_uv
                    lambdas_out[v] -= lambda_uv
        else:
            for u, v in uv_pairs_u_greater_v:
                lambda_uv = self.get_single_lambda_uv(u, v, s_uvs, scores)
                lambdas_out[u] += lambda_uv
                lambdas_out[v] -= lambda_uv
        return lambdas_out
        
    def train_once(self, X_train, query, labels):
        scores = self.score(query).flatten()[:len(labels)]
        qid = query.get_qid()
        if self.model_type != 'Pointwise':
            lambdas = self.compute_lambdas(labels, scores, 
                                           self.uv_pairs[qid], 
                                           self.s_uv_sets[qid],
                                           self.norming_dcgs[qid])
            lambdas = np.resize(lambdas, self.batch_size)
        labels = np.resize(labels, self.batch_size)
        X_train = np.resize(X_train, (self.batch_size, self.feature_count))
        if self.model_type == 'Pointwise':
            batch_train_loss = self.iter_funcs['train'](X_train, labels)
        else:
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
        return batch_train_loss

    def train(self):
        X_trains = self.train_queries.get_feature_vectors()
        queries = self.train_queries.values()
        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in xrange(len(queries)):
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()
                batch_train_loss = self.train_once(X_trains[random_index],
                                                   queries[random_index], 
                                                    labels)
                batch_train_losses.append(batch_train_loss)
            avg_train_loss = np.mean(batch_train_losses)
            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }

# Function to get NDCG score of a model 'l_rank' trying to predict relevance 
# scores for a set of queries 'qry_set'
def get_ndcg(qry_set, l_rank):
    qids = qry_set.get_qids()
    ndcg = np.zeros(len(qids))
    for q in range(len(qids)):
        test_query = qry_set.get_query(qids[q])
        #print sum(test_query.get_labels())
        scores = l_rank.score(test_query)
        scores = np.array([scores[i][0] for i in range(len(scores))])
        order = np.argsort(scores)
        ndcg[q] = ndcg_at_k(test_query.get_labels()[order], 10)
        #print "ndcg = " + str(ndcg[q])
    return np.mean(ndcg)