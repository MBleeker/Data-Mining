import csv
import cPickle
import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from itertools import count
import query

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
MOMENTUM = 0.95
SIGMA = 1

class LambdaRank:

    NUM_INSTANCES = count()

    def __init__(self, feature_count, mod_type = 'Pointwise', learn = 0.001):
        self.lr = learn
        self.model_type = mod_type
        self.feature_count = feature_count
        self.output_layer = self.build_model(feature_count,1,BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)

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
    def train_with_queries(self, train_queries, num_epochs):
        self.uv_pairs = {}
        self.s_uv_sets = {}
        for q in train_queries.get_qids():
            self.uv_pairs_u_greater_v = []
            labels = train_queries.get_query(q).get_labels()
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if labels[i] > labels[j]:
                        self.uv_pairs_u_greater_v.append([i, j])
            self.uv_pairs[q] = self.uv_pairs_u_greater_v
            self.s_uv_sets[q] = self.get_suvs(labels)        
        
        try:
            now = time.time()
            for epoch in self.train(train_queries):
                if epoch['number'] % 10 == 0:
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


    def build_model(self,input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print "input_dim",input_dim, "output_dim",output_dim
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_in,
            num_units=60,
            nonlinearity=lasagne.nonlinearities.sigmoid,
        )

        l_hidden2 = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=30,
            nonlinearity=lasagne.nonlinearities.sigmoid,
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden2,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.sigmoid,
        )

        return l_out
    
    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer, pointwise = False,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE, momentum=MOMENTUM,
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
            loss_train = lasagne.objectives.squared_error(-output, y_batch)
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

        print "finished create_iter_functions"
        return dict(
            train=train_func,
            out=score_func,
        )

    def get_dcg_with_labels_switched(self, dcg_base, labels, u, v):
        a = 2 ** (labels[v] - 1)
        b = 2 ** (labels[u] - 1)
        return (a - b) / np.log2(2 + u) + (b - a) / np.log2(2 + v)
        
    def get_single_lambda_uv(self, u, v, s_uvs, scores):
        return -1. * SIGMA * (0.5 * (1. - s_uvs[u, v]) - \
                              1. / (1. + np.exp(SIGMA * (scores[v] - scores[u]))))
        
    def get_single_lambda_rank_uv(self, u, v, s_uvs, scores):
        # Equation 6 in Burges paper
        return -1. * SIGMA / (1. + np.exp(SIGMA * (scores[v] - scores[u])))
        
    def compute_lambdas(self, labels, scores, uv_pairs_u_greater_v, s_uvs):
        lambdas_out = np.zeros(len(labels))
        if self.model_type == 'LambdaRank':
            normalising_dcg = dcg_at_k(sorted(labels, reverse=True), len(labels))
            if normalising_dcg != 0.:
                dcg_base = dcg_at_k(labels, len(labels))
                for u, v in uv_pairs_u_greater_v:
                    ndcg_chg = self.get_dcg_with_labels_switched(dcg_base, labels, u, v)
                    ndcg_chg -= dcg_base
                    ndcg_chg /= normalising_dcg
                    # not taking abs value ndcg_chg seems to be the way to go....
                    lambda_uv = self.get_single_lambda_rank_uv(u, v, s_uvs, scores) * ndcg_chg
                    lambdas_out[u] += lambda_uv
                    lambdas_out[v] -= lambda_uv
            return lambdas_out
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
            lambdas = self.compute_lambdas(labels, scores, self.uv_pairs[qid], 
                                           self.s_uv_sets[qid])
            lambdas = np.resize(lambdas, BATCH_SIZE)
        labels = np.resize(labels, BATCH_SIZE)
        X_train = np.resize(X_train, (BATCH_SIZE, self.feature_count))
        if self.model_type == 'Pointwise':
            batch_train_loss = self.iter_funcs['train'](X_train, labels)
        else:
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
        return batch_train_loss

    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()
        queries = train_queries.values()
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