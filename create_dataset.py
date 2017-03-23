
# coding: utf-8

# In[ ]:

import cv2                
import numpy as np        
import os                 
from random import shuffle 
from tqdm import tqdm   
import theano
from theano import tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, relu
import timeit
import pickle as pickle
import os
import sys
import gzip
import random

TRAIN_DIR = 'train - Copy'
TEST_DIR = 'test'
IMG_SIZE = 50
LR = 1e-3


def create_dataset():

    print('create training set')

    def image_label(img):
        word_label = img.split('.')[0]

        if word_label[0] == 'c': return 1

        elif word_label[0] == 'd': return 0

    def create_train_data():
        training_data = []
        labels = []
        for img in tqdm(os.listdir(TRAIN_DIR)):
            label = image_label(img)
            path = os.path.join(TRAIN_DIR,img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            x = np.array(img).flatten().T/255
            y = label
            training_data.append([x,y])
        random.shuffle(training_data)
        np.save('train_data4.npy', training_data)
        return training_data

    train_data = create_train_data()

    train_set = []
    valid_set = []
    test_set = []


    for i in range(len(train_data)):
    	if i < 0.8*len(train_data):
	     train_set.append(train_data[i])
    	if i>= 0.8*len(train_data) and i< 0.85*len(train_data):
	     valid_set.append(train_data[i])
    	if i>= 0.85*len(train_data):
	     test_set.append(train_data[i])



    train_x = []
    train_y = []
    test_x = []
    test_y = []
    valid_x = []
    valid_y = []

    for i in range(len(train_set)):

        train_x.append(train_set[i][0])
        train_y.append(train_set[i][1])

    for i in range(len(test_set)):

        test_x.append(test_set[i][0])
        test_y.append(test_set[i][1])

    for i in range(len(valid_set)):

        valid_x.append(valid_set[i][0])
        valid_y.append(valid_set[i][1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)


    #### function obtained from previous code #####
    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """

        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_x, test_y)
    valid_set_x, valid_set_y = shared_dataset(valid_x, valid_y)
    train_set_x, train_set_y = shared_dataset(train_x, train_y)


    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
        (test_set_x, test_set_y)]

    return rval


