import time
import numpy as np
import theano
import theano.tensor as T
import lasagne

class NNModel:
    def __init__(self, batchsize=500, learnrate=0.01, momentum=0.9, 
                 num_epochs=10, depth=2, width=[10,10], drop_input=.2, 
                 drop_hidden=.5, normalise = False, report_every=10):
        self.width = width
        self.drop_input = drop_input
        self.drop_hidden = drop_hidden
        self.depth = depth
        self.batch_size = batchsize
        self.report_every = report_every
        self.num_epochs = num_epochs
        self.learning_rate = learnrate
        self.momentum = momentum
        self.normalise = normalise
        self.fitted = False
        self.initialised = False
        
    def build_custom_mlp(self, input_var=None):
        network = lasagne.layers.InputLayer(shape=(None, self.input_shape),
                                            input_var=input_var)
        if self.drop_input:
            network = lasagne.layers.dropout(network, p=self.drop_input)
        # Hidden layers and dropout:
        #nonlin = lasagne.nonlinearities.tanh
        hidden_nonlin = lasagne.nonlinearities.leaky_rectify
        output_nonlin=lasagne.nonlinearities.linear
        for layer in range(len(self.width)):
            network = lasagne.layers.DenseLayer(
                    network, self.width[layer], nonlinearity=hidden_nonlin)
            if self.drop_hidden:
                network = lasagne.layers.dropout(network, p=self.drop_hidden)
        # Output layer:
        network = lasagne.layers.DenseLayer(network, 
                                    1, nonlinearity=output_nonlin)
        return network
    
    def iterate_minibatches(self, inputs, targets, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - self.batch_size + 1, 
                               self.batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            yield inputs[excerpt], targets[excerpt]
            
    def normalise_X(self, X, reset = False):
        if reset:
            self.x_mean_train = np.mean(X, axis = 0)
            self.x_std_train = np.std(X, axis = 0)
        X = X - self.x_mean_train
        X = X / self.x_std_train
        return X
        
    def normalise_y(self, y, reset = False):
        if reset:
            self.y_mean_train = np.mean(y)
            self.y_std_train = np.std(y)
        y = y - self.y_mean_train
        y = y / self.y_std_train
        return y
        
    def denormalise_y(self, preds):
        return self.y_mean_train + self.y_std_train * preds
    
    def initialise_model(self, X_train, y_train):
        print 'Initialising model...'
        self.input_shape = X_train.shape[1]
        input_var = T.matrix('inputs')
        target_var = T.matrix('targets')
        
        if self.normalise:
            y_train = self.normalise_y(y_train, reset = True)
            X_train = self.normalise_X(X_train, reset = True)
    
        # Create neural network model
        self.network = self.build_custom_mlp(input_var)
        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, 
                                            learning_rate=self.learning_rate, 
                                            momentum=self.momentum)
        test_prediction = lasagne.layers.get_output(self.network,
                                                    deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_prediction,
                                                     target_var)
        test_loss = test_loss.mean()
        self.train_fn = theano.function([input_var, target_var], loss, 
                                   updates=updates, allow_input_downcast=True)
        self.predict_output = theano.function([input_var],
                                              outputs=test_prediction,
                                              allow_input_downcast=True)
        self.initialised = True
    
    def get_norm_constants(self):
        return self.x_mean_train, self.x_std_train, \
               self.y_mean_train, self.y_std_train
    
    def set_norm_constants(self, xm, xs, ym, ys):
        self.x_mean_train, self.x_std_train, \
            self.y_mean_train, self.y_std_train = xm, xs, ym, ys
    
    def fit(self, X_train, y_train):
        if not self.initialised:
            self.initialise_model(X_train, y_train)
        self.train(X_train, y_train)
        
    def train(self, X_train, y_train, epochs = None):
        if not self.initialised:
            self.initialise_model(X_train, y_train)
        y_train = y_train.reshape(y_train.shape[0], 1)
        if self.normalise:
            y_train = self.normalise_y(y_train)
            X_train = self.normalise_X(X_train)
        # Finally, launch the training loop.
        print('Starting training...')
        # We iterate over epochs:
        if epochs != None:
            self.num_epochs = epochs
        start_time = time.time()
        for epoch in range(self.num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            for batch in self.iterate_minibatches(X_train, y_train, 
                                                  shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1
    
            if epoch % self.report_every == 0:
                # Then we print the results for this epoch:
                print('Epoch {} of {} took {:.3f}s'.format(epoch + 1, 
                                                           self.num_epochs,
                                (time.time() - start_time)/self.report_every))
                print('  training loss:\t\t{:.6f}'.\
                    format(train_err / train_batches))
                start_time = time.time()
        self.fitted = True    
        
    def predict(self, X):
        if self.fitted:
            if self.normalise:
                X = self.normalise_X(X)
            return self.denormalise_y(self.predict_output(X))
        else:
            print 'No model fitted yet to predict with!'
            return None