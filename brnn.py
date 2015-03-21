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

        self.output = T.cast(self.activation_fn(T.dot(inputs, self.W) + self.b) * rng.binomial(size=(output_size,), p=1.0-dropout_rate), dtype=theano.config.floatX)
        self.params = [self.W, self.b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def set_parameters(self, parameters):
        self.W.set_value(parameters['W'])
        self.b.set_value(parameters['b'])


class RecurrentLayer:
    def __init__(self, inputs, input_size, output_size, is_backward=False, parameters=None):

        if parameters is None:
            self.W_if = U.create_shared(U.initial_weights(input_size, output_size), name='W_if')
            self.W_ff = U.create_shared(U.initial_weights(output_size, output_size), name='W_ff')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W_if = theano.shared(parameters['W_if'], name='W_if')
            self.W_ff = theano.shared(parameters['W_ff'], name='W_ff')
            self.b = theano.shared(parameters['b'], name='b')

        initial = T.zeros((output_size,))
        self.is_backward = is_backward
        self.activation_fn = lambda x: T.cast(T.minimum(x * (x > 0), 20), dtype='float32')#dtype=theano.config.floatX)

        self.output, _ = theano.scan(
            lambda in_t, out_tminus1: self.activation_fn(T.dot(out_tminus1, self.W_ff) + T.dot(in_t, self.W_if) + self.b),
            sequences=[inputs],
            outputs_info=[initial],
            go_backwards=self.is_backward
        )

        self.params = [self.W_if, self.W_ff, self.b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def set_parameters(self, parameters):
        self.W_if.set_value(parameters['W_if'])
        self.W_ff.set_value(parameters['W_ff'])
        self.b.set_value(parameters['b'])


class SoftmaxLayer:
    def __init__(self, inputs, input_size, output_size, parameters=None):

        if parameters is None:
            self.W = U.create_shared(U.initial_weights(input_size, output_size), name='W')
            self.b = U.create_shared(U.initial_weights(output_size), name='b')
        else:
            self.W = theano.shared(parameters['W'], name='W')
            self.b = theano.shared(parameters['b'], name='b')

        self.output = T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        self.params = [self.W, self.b]

    def get_parameters(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def set_parameters(self, parameters):
        self.W.set_value(parameters['W'])
        self.b.set_value(parameters['b'])


# Courtesy of https://github.com/rakeshvar/rnn_ctc
# With T.eye() removed for k!=0 (not implemented on GPU for k!=0)
class CTCLayer():
    def __init__(self, inpt, labels, blank):
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
        n_labels = labels.shape[0]

        big_I = T.cast(T.eye(n_labels+2), 'float64')
        recurrence_relation1 = T.cast(T.eye(n_labels), 'float64') + big_I[2:,1:-1] + big_I[2:,:-2] * T.cast((T.arange(n_labels) % 2), 'float64')
        recurrence_relation = T.cast(recurrence_relation1, 'float64')

        '''
        Forward path probabilities
        '''
        pred_y = inpt[:, labels]

        probabilities, _ = theano.scan(
            lambda curr, prev: curr * T.dot(prev, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.cast(T.eye(n_labels)[0], 'float64')]
        )

        # Final Costs
        labels_probab = T.sum(probabilities[-1, -2:])
        self.cost = -T.log(labels_probab)
        self.params = []


class BRNN:
    def __init__(self, input_dimensionality, output_dimensionality, params=None, learning_rate=0.01, momentum=.25):
        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed=1234)

        input_seq = T.fmatrix('input_seq')
        label_seq = T.ivector('label')
        dropoutRate = T.fscalar('dropoutRate')      # set to 0.0 during non-training

        if params is None:
            self.ff1 = FeedForwardLayer(input_seq, self.input_dimensionality, 2048, rng=srng, dropout_rate=dropoutRate)
            self.ff2 = FeedForwardLayer(self.ff1.output, 2048, 2048, rng=srng, dropout_rate=dropoutRate)
            self.ff3 = FeedForwardLayer(self.ff2.output, 2048, 2048, rng=srng, dropout_rate=dropoutRate)
            self.rf = RecurrentLayer(self.ff3.output, 2048, 1024, False)     # Forward layer
            self.rb = RecurrentLayer(self.ff3.output, 2048, 1024, True)      # Backward layer

            # REVERSE THE BACKWARDS RECURRENT OUTPUTS IN TIME (from [T-1, 0] ===> [0, T-1]
            self.s = SoftmaxLayer(T.concatenate((self.rf.output, self.rb.output[::-1, :]), axis=1), 2*1024, self.output_dimensionality)

        else:
            self.ff1 = FeedForwardLayer(input_seq, self.input_dimensionality, 2048, parameters=params[0], rng=srng, dropout_rate=dropoutRate)
            self.ff2 = FeedForwardLayer(self.ff1.output, 2048, 2048, parameters=params[1], rng=srng, dropout_rate=dropoutRate)
            self.ff3 = FeedForwardLayer(self.ff2.output, 2048, 2048, parameters=params[2], rng=srng, dropout_rate=dropoutRate)
            self.rf = RecurrentLayer(self.ff3.output, 2048, 1024, False, parameters=params[3])     # Forward layer
            self.rb = RecurrentLayer(self.ff3.output, 2048, 1024, True, parameters=params[4])      # Backward layer

            # REVERSE THE BACKWARDS RECURRENT OUTPUTS IN TIME (from [T-1, 0] ===> [0, T-1]
            self.s = SoftmaxLayer(T.concatenate((self.rf.output, self.rb.output[::-1, :]), axis=1), 2*1024, self.output_dimensionality, parameters=params[5])


        ctc = CTCLayer(self.s.output, label_seq, self.output_dimensionality-1)
        l2 = T.sum(self.ff1.W**2) + T.sum(self.ff2.W**2) + T.sum(self.ff3.W**2) + T.sum(self.s.W**2) + T.sum(self.rf.W_if**2) + T.sum(self.rf.W_ff**2) + T.sum(self.rb.W_if**2)
        
        updates = []
        for layer in (self.s, self.rb, self.rf, self.ff3, self.ff2, self.ff1):
            for p in layer.params:
                param_update = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
                grad = T.grad(ctc.cost + .005*l2, p)
                updates.append((p, p - learning_rate * param_update))
                updates.append((param_update, momentum * param_update + (momentum) * grad))
                #updates.append((p, p - T.cast(learning_rate, dtype='float32')*T.grad(ctc.cost, p)))

        self.trainer = theano.function(
            inputs=[input_seq, label_seq, dropoutRate],
            #outputs=[ctc.cost]
            outputs=[ctc.cost],
            updates=updates
        )
        self.tester = theano.function(
            inputs=[input_seq, label_seq, dropoutRate],
            outputs=[ctc.cost, self.s.output]
        )


    def dump(self, f_path):
        f = file(f_path, 'wb')
        for obj in [self.ff1.get_parameters(), self.ff2.get_parameters(), self.ff3.get_parameters(), self.rf.get_parameters(), self.rb.get_parameters(), self.s.get_parameters()]:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

class Network:
    def __init__(self):
        self.nn = None

    def create_network(self, input_dimensionality, output_dimensionality, learning_rate=0.01, momentum=.25):
        self.nn = BRNN(input_dimensionality, output_dimensionality, params=None, learning_rate=learning_rate, momentum=momentum)
        return self.nn

    def load_network(self, path, input_dimensionality, output_dimensionality, learning_rate=0.01, momentum=.25):
        f = file(path, 'rb')
        parameters = []
        for i in np.arange(6):
            parameters.append(pickle.load(f))
        f.close()

        for p in parameters:
            print type(p)

        self.nn = BRNN(input_dimensionality, output_dimensionality, params=parameters, learning_rate=learning_rate, momentum=momentum)
        return self.nn

    def dump_network(self, path):
        if self.nn is None:
            return False

        self.nn.dump(path)

    def get_network(self):
        assert(self.nn is not None)
        return [self.nn.ff1.get_parameters(), self.nn.ff2.get_parameters(), self.nn.ff3.get_parameters(), self.nn.rf.get_parameters(), self.nn.rb.get_parameters(), self.nn.s.get_parameters()]

    def set_network(self, parameters):
        assert(type(parameters) == list)
        assert(len(parameters) == 6)

        self.nn.ff1.set_parameters(parameters[0])
        self.nn.ff2.set_parameters(parameters[1])
        self.nn.ff3.set_parameters(parameters[2])
        self.nn.rf.set_parameters(parameters[3])
        self.nn.rb.set_parameters(parameters[4])
        self.nn.s.set_parameters(parameters[5])
