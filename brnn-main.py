__author__ = 'pat'
import numpy as np
import math
import time
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cPickle as pickle
import theano

import brnn as nn


def str_to_seq(str):
    seq = []
    for c in str:
        val = ord(c)
        if val==32:
            val = 26
        elif val==39:
            val=27
        elif val==45:
            val=28
        else:
            val-=97
        seq.append(val)
    return seq
    

def seq_to_str(seq):
    str = ''
    for elem in seq:
        if elem==26:
            str += ' '
        elif elem==27:
            str += '\''
        elif elem==28:
            str += '-'
        elif elem==29:
            pass
        else:
            str += chr(elem+97)
    return str
    

# Remove consecutive symbols and blanks
def F(pi, blank):
    return [a for a in [key for key, _ in groupby(pi)] if a != blank]
    

# Insert blanks between unique symbols, and at the beginning and end
def make_l_prime(l, blank):
    result = [blank] * (len(l) * 2 + 1)
    result[1::2] = l
    return result
    
    
def log_it(f, epoch=None, error=None, etime=None, samples=None, nan=False, etc=None):
    s5 = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
    if epoch is not None:
        message = time.strftime('%H:%M:%S') + '\tEpoch: ' + str(epoch) + '\tAvg. Error: ' + str(error / samples) + '\tin ' + str(etime) + 's\t' +str(samples) + ' samples\tSamples/sec: ' + str(samples/etime) + '\tApprox. Speed: ' + str(samples * 5 / etime) + 'x real-time'
        print '\n', message, '\n'
        f.write(message.replace('\t', s5) + '<br />')
    elif nan:
        print time.strftime('%H:%M:%S') + '\n========\nHIT NAN=======\nInvalidating Epoch\n===========\n'
        f.write('<br />' + time.strftime('%H:%M:%S') + '\n========\nHIT NAN=======\nInvalidating Epoch\n===========\n\n<br />')
        
    else:
        print etc
        f.write(time.strftime('%H:%M:%S') +s5+ etc.replace('\n', '<br />').replace('\t', s5) + '<br />')


alphabet = np.arange(29) #[a,....,z, space, ', -]


# Load samples
f = open('win3_l35.pkl', 'rb')
samples = pickle.load(f)
f.close()

samples = samples[:1000]

visual_samples = [samples[0], samples[-1]]


# HYPER-PARAMETERS
learning_rate = .01
momentum_coefficient = .5
dropout_rate = .1

# Automatically Generated Parameters
alphabet_len = len(alphabet)
input_dim = 560


net = nn.Network()      # Network wrapper
rng = np.random.RandomState(int(time.time()))

# create network
try:
    last_good = -1
    #log_file = file('/var/www/html/status.html', 'a')
    log_file = file('status.html', 'a')
    duration = time.time()
    if last_good == -1:
        network = net.create_network(input_dim, alphabet_len+1, learning_rate=learning_rate, momentum=momentum_coefficient)
        log_it(log_file, etc='created Network - num samples:' + str(len(samples)) + '\tDuration: ' + str(time.time()-duration))
    else:
        picklePath = 'saved_models/' + str(last_good) + '.pkl'
        print 'loading from', picklePath
        network = net.load_network(picklePath, 560, len(alphabet)+1, .0005, .75)        #x3 for the window
        log_it(log_file, etc='loaded Network - num samples:' + str(len(samples)) + '\tDuration: ' + str(time.time()-duration))
    log_file.close()

    # Start a new Epoch
    for epoch in np.arange(last_good+1, 100000):

        rng.shuffle(samples)
        prev_model = net.get_network()

        error_avg_epoch = 0.0
        duration = time.time()
        for sample in samples:
            error_avg_epoch += network.trainer(sample[1], sample[0], dropout_rate)[0]

            if error_avg_epoch == np.nan or error_avg_epoch == np.inf:
                net.set_network(prev_model)
                print "hit nan/inf"
                break

        if not (error_avg_epoch == np.nan or error_avg_epoch == np.inf):

            print 'Epoch:\t%.3fs\t' % (epoch, time.time()-duration)

            for sample in visual_samples:
                cst, out = network.tester(sample[1], sample[0], 0.0)

                print '\t%.5f\t' % (cst), seq_to_str(sample[0]), ' || ', seq_to_str(out)

except KeyboardInterrupt:
    pass