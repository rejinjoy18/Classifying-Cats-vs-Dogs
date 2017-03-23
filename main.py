
# coding: utf-8

# In[ ]:




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



from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression, load_data
from Conv_Pool import LeNetConvPoolLayer
from create_dataset import create_dataset

## Number of kernels for each layer set to 32, 64 and 128 respectively

def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    nkerns=[32, 64, 128], batch_size=300):
  
    rng = np.random.RandomState(23455)

    datasets = create_dataset()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 100 * 100)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer

    layer0_input = x.reshape((batch_size, 1, 100, 100))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (100-5+1 , 100-5+1) = (96, 96)
    # maxpooling reduces this further to (96/2, 96/2) = (48, 48)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 48, 48)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 100, 100),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2), Pool = True
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (48-5+1, 48-5+1) = (44, 44)
    # maxpooling reduces this further to (44/2, 44/2) = (22, 22)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 22, 22)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 48, 48),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2), Pool = True
    )
    
    
    # Construct the third convolutional pooling layer
    # filtering reduces the image size to (22-3+1, 22-3 +1) = (20, 20)
    # Since "Pool" is set to False, pooling operation will not take place
    
    layer2 = LeNetConvPoolLayer(
        rng, 
        input = layer1.output, 
        image_shape = (batch_size, nkerns[1], 22, 22),
        filter_shape = (nkerns[2], nkerns[1], 3, 3),
        Pool = False
    )
    
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 20 * 20),
    # or (30, 128 * 20 * 20) with the default values.
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 20 * 20,
        n_out=batch_size,
        activation=relu
    )

    # classify the values of the fully-connected sigmoidal layer (5 classes)
    layer4 = LogisticRegression(input=layer3.output, n_in=batch_size, n_out=5)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    
    #### updating the parameters using SGD with momentum
    updates = []
    momentum = 0.9
    
    for param in params:
        update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate*update))
        updates.append((update, momentum*update + (1. - momentum)*T.grad(cost, param)))
     

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
 
 ### decaying learning rate after every 10 epochs and decaying momentum value after every 10 epochs

        if (epoch%10 == 0):
            learning_rate -= 0.05*learning_rate
            print("New learning rate is ", learning_rate)
            
            momentum -= 0.05*momentum
            print("New momentum is ", momentum)
        
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *                         improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code ' +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)

