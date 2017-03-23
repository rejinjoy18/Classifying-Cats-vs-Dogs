
# coding: utf-8

# In[ ]:

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, relu



class LeNetConvPoolLayer(object):
    
### added boolean "Pool" parameter to check whether pooling has to be peformed 
    
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2), Pool = True):
        
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        
        W_bound = np.sqrt(6./ (fan_in + fan_out))
        
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype = theano.config.floatX), borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype = theano.config.floatX)
        self.b = theano.shared(value = b_values, borrow=True)
        
        conv_out = conv2d(input=input, filters=self.W, filter_shape = filter_shape, input_shape = image_shape)

### if statement to check whether pooling must be performed        
        if Pool == True:
            conv_out = pool.pool_2d(input = conv_out, ds = poolsize, ignore_border = True)

### activation fn changed to relu
        self.output = relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        self.params = [self.W, self.b]
        
        self.input = input

