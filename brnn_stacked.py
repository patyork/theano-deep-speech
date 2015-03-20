__author__ = 'pat'
'''
Bidirectional Recurrent Neural Network
with Connectionist Temporal Classification (CTC)
  courtesy of https://github.com/shawntan/rnn-experiment
  courtesy of https://github.com/rakeshvar/rnn_ctc
implemented in Theano and optimized for use on a GPU
'''

import theano
import theano.tensor as T
from theano_toolkit import utils as U
from theano_toolkit import updates
import numpy as np
import cPickle as pickle
import time

#THEANO_FLAGS='device=cpu,floatX=float32'
#theano.config.warn_float64='warn'

#theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity='high'
theano.config.on_unused_input='warn'


class FeedForwardLayer:
    def __init__(self, inputs, input_size, output_size, rng, dropout_rate, parameters=None):
        self.activation_fn = lambda x: T.minimum(x * (x > 0), 20)

        if parameters is None:
            self.W = U.create_shared(U.initial_weights(input_size, output_size), name='W')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W = theano.shared(parameters['W'], name='W')
            self.b = theano.shared(parameters['b'], name='b')

        self.output = T.cast(self.activation_fn(T.dot(inputs, self.W) + self.b) * rng.binomial(size=(output_size,), p=1.0-dropout_rate), dtype='float32')

        self.params = [self.W, self.b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params



class RecurrentLayer:
    def __init__(self, inputs, input_size, output_size, batch_size, is_backward=False, parameters=None):

        if parameters is None:
            self.W_if = U.create_shared(U.initial_weights(input_size, output_size), name='W_if')
            self.W_ff = U.create_shared(U.initial_weights(output_size, output_size), name='W_ff')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W_if = theano.shared(parameters['W_if'], name='W_if')
            self.W_ff = theano.shared(parameters['W_ff'], name='W_ff')
            self.b = theano.shared(parameters['b'], name='b')

        initial = T.zeros((batch_size, output_size))
        self.is_backward = is_backward
        self.activation_fn = lambda x: T.cast(T.minimum(x * (x > 0), 20), dtype='float32')#dtype=theano.config.floatX)


        # Unstack the inputs into indivisual samples
        inputs = inputs.reshape((batch_size, inputs.shape[0]/batch_size, inputs.shape[1]))    # (batch size, input shape[0]/batch size, input shape[1])

        # Swap axes to loop over individual time steps (over the entire batch)
        inputs = T.swapaxes(inputs, 0, 1)

        def step(in_t, out_tminus1):
            return self.activation_fn(T.dot(out_tminus1, self.W_ff) + T.dot(in_t, self.W_if) + self.b)

        output, _ = theano.scan(
            step,
            sequences=[inputs],
            outputs_info=[initial],
            go_backwards=self.is_backward
        )
        self.output = T.swapaxes(output, 0, 1)         # Swap axis to get back to (batch size, time step, output)

        self.params = [self.W_if, self.W_ff, self.b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params


class SoftmaxLayer:
    def __init__(self, inputs, input_size, output_size, parameters=None):

        if parameters is None:
            self.W = U.create_shared(U.initial_weights(input_size, output_size), name='W')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W = theano.shared(parameters['W'], name='W')
            self.b = theano.shared(parameters['b'], name='b')

        self.output, _ = theano.scan(
            lambda x: T.nnet.softmax(T.dot(x, self.W) + self.b),
            sequences=[inputs]
        )
        self.params = [self.W, self.b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params



# Courtesy of https://github.com/rakeshvar/rnn_ctc
# With T.eye() removed for k!=0 (not implemented on GPU for k!=0)
class CTCLayer():
    def __init__(self, inpt, labels, blank, batch_size):
        '''
        Recurrent Relation:
        A matrix that specifies allowed transistions in paths.
        At any time, one could
        0) Stay at the same label (diagonal is identity)
        1) Move to the next label (first upper diagonal is identity)
        2) Skip to the next to next label if
            a) next label is blank and
            b) the next to next label is different from the current
            (Second upper diagonal is product of conditons a & b)
        '''
        n_labels = labels[0].shape[0]
        big_I = T.cast(T.eye(n_labels+2), 'float64')
        recurrence_relation = T.cast(T.eye(n_labels), 'float64') + big_I[2:,1:-1] + big_I[2:,:-2] * T.cast((T.arange(n_labels) % 2), 'float64')
        recurrence_relation = T.cast(recurrence_relation, 'float64')


        #inpt = T.reshape(inpt, (batch_size, inpt.shape[0]/batch_size, 30))

        def step(input, label):
            '''
            Forward path probabilities
            '''
            pred_y = input[:, label]

            probabilities, _ = theano.scan(
                lambda curr, prev: curr * T.dot(prev, recurrence_relation),
                sequences=[pred_y],
                outputs_info=[T.cast(T.eye(n_labels)[0], 'float64')]
            )
            return -T.log(T.sum(probabilities[-1, -2:]))

        probs, _ = theano.scan(
            step,
            sequences=[inpt, labels]
        )

        self.cost = T.cast(T.mean(probs), dtype='float32')
        self.params = []


class BRNN:
    def __init__(self, input_dimensionality, output_dimensionality, params=None, batch_size=100, learning_rate=0.01, momentum=.25):
        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed=1234)

        input_stack = T.fmatrix('input_seq')
        label_stack = T.imatrix('label')
        dropoutRate = T.fscalar('dropoutRate')      # set to 0.0 during non-training

        if params is None:
            self.ff1 = FeedForwardLayer(input_stack, self.input_dimensionality, 2048, rng=srng, dropout_rate=dropoutRate)
            self.ff2 = FeedForwardLayer(self.ff1.output, 2048, 2048, rng=srng, dropout_rate=dropoutRate)
            self.ff3 = FeedForwardLayer(self.ff2.output, 2048, 2048, rng=srng, dropout_rate=dropoutRate)
            self.rf = RecurrentLayer(self.ff3.output, 2048, 1024, batch_size, False)     # Forward layer
            self.rb = RecurrentLayer(self.ff3.output, 2048, 1024, batch_size, True)      # Backward layer

            # REVERSE THE BACKWARDS RECURRENT OUTPUTS IN TIME (from [T-1, 0] ===> [0, T-1]
            self.s = SoftmaxLayer(T.concatenate((self.rf.output, self.rb.output[:, ::-1, :]), axis=2), 2*1024, self.output_dimensionality)

        else:
            self.ff1 = FeedForwardLayer(input_stack, self.input_dimensionality, 2048, parameters=params[0], rng=srng, dropout_rate=dropoutRate)
            self.ff2 = FeedForwardLayer(self.ff1.output, 2048, 2048, parameters=params[1], rng=srng, dropout_rate=dropoutRate)
            self.ff3 = FeedForwardLayer(self.ff2.output, 2048, 2048, parameters=params[2], rng=srng, dropout_rate=dropoutRate)
            self.rf = RecurrentLayer(self.ff3.output, 2048, 1024, False, parameters=params[3])     # Forward layer
            self.rb = RecurrentLayer(self.ff3.output, 2048, 1024, True, parameters=params[4])      # Backward layer

            # REVERSE THE BACKWARDS RECURRENT OUTPUTS IN TIME (from [T-1, 0] ===> [0, T-1]
            self.s = SoftmaxLayer(T.concatenate((self.rf.output, self.rb.output[:, ::-1, :]), axis=2), 2*1024, self.output_dimensionality, parameters=params[5])


        ctc = CTCLayer(self.s.output, label_stack, self.output_dimensionality-1, batch_size)
        
        updates = []
        for layer in (self.s, self.rb, self.rf, self.ff3, self.ff2, self.ff1):
            for p in layer.params:
                #param_update = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
                #grad = T.grad(ctc.cost, p)
                #updates.append((p, p - learning_rate * param_update))
                #updates.append((param_update, momentum * param_update + (1. - momentum) * grad))
                updates.append((p, p - T.cast(learning_rate, dtype='float32')*T.grad(ctc.cost, p)))

        self.trainer = theano.function(
            inputs=[input_stack, label_stack, dropoutRate],
            #outputs=[ctc.cost]
            outputs=[ctc.cost],
            updates=updates
        )


    def dump(self, f_path):
        f = file(f_path, 'wb')
        for obj in [self.ff1.get_parameters(), self.ff2.get_parameters(), self.ff3.get_parameters(), self.rf.get_parameters(), self.rb.get_parameters(), self.s.get_parameters()]:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

class Network:
    def __init__(self):
        self.nn = None

    def create_network(self, input_dimensionality, output_dimensionality, batch_size=50, learning_rate=0.01, momentum=.25):
        self.nn = BRNN(input_dimensionality, output_dimensionality, params=None, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
        return self.nn

    def load_network(self, path, input_dimensionality, output_dimensionality, batch_size=50, learning_rate=0.01, momentum=.25):
        f = file(path, 'rb')
        parameters = []
        for i in np.arange(6):
            parameters.append(pickle.load(f))
        f.close()

        for p in parameters:
            print type(p)

        self.nn = BRNN(input_dimensionality, output_dimensionality, params=parameters, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
        return self.nn

    def dump_network(self, path):
        if self.nn is None:
            return False

        self.nn.dump(path)
