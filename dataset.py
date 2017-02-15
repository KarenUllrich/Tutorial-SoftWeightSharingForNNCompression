#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loading the MNIST in the right format.

Implementation is close to [1].

Karen Ullrich, Jan 2017

      ... [1] [Keras Tutorial on CNNs](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)
"""

from numpy import transpose

from keras import backend as K
from keras.datasets import mnist as MNIST
from keras.utils import np_utils


def mnist():
    img_rows, img_cols = 28, 28
    nb_classes = 10
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = MNIST.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

    if K._BACKEND == "tensorflow":
        X_train = transpose(X_train, axes=[0, 2, 3, 1])
        X_test = transpose(X_test, axes=[0, 2, 3, 1])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print("Successfully loaded %d train samples and %d test samples." % (X_train.shape[0], X_test.shape[0]))

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return [X_train, X_test], [Y_train, Y_test], [img_rows, img_cols], nb_classes
