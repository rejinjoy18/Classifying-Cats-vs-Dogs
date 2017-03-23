
# coding: utf-8

# In[ ]:

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, relu
import timeit
import pickle as pickle
import os
import sys
import gzip

class HiddenLayer(object):
    
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=relu):
        self.input = input
        
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
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

