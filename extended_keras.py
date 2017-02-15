#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Methods to compliment the keras engine

Karen Ullrich, Jan 2017
"""
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import imageio

import keras
from keras import backend as K

from helpers import special_flatten


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

def collect_trainable_weights(layer):
    """Collects all `trainable_weights` attributes,
    excluding any sublayers where `trainable` is set the `False`.
    """
    trainable = getattr(layer, 'trainable', True)
    if not trainable:
        return []
    weights = []
    if layer.__class__.__name__ == 'Sequential':
        for sublayer in layer.flattened_layers:
            weights += collect_trainable_weights(sublayer)
    elif layer.__class__.__name__ == 'Model':
        for sublayer in layer.layers:
            weights += collect_trainable_weights(sublayer)
    else:
        weights += layer.trainable_weights
    # dedupe weights
    weights = list(set(weights))
    # TF variables have auto-generated the name, while Theano has auto-generated the auto_name variable.
    # name in Theano is sometimes None.
    # However, to work save_model() and load_model() properly, weights must be sorted by names.
    if weights:
        if "theano" == K.backend():
            weights.sort(key=lambda x: x.name if x.name else x.auto_name)
        else:
            weights.sort(key=lambda x: x.name)
    return weights


def extract_weights(model):
    """Extract symbolic, trainable weights from a Model."""
    trainable_weights = []
    for layer in model.layers:
        trainable_weights += collect_trainable_weights(layer)
    return trainable_weights


# ---------------------------------------------------------
# objectives
# ---------------------------------------------------------

def identity_objective(y_true, y_pred):
    """Hack to turn Keras' Layer engine into an empirical prior on the weights"""
    return y_pred


# ---------------------------------------------------------
# logsumexp
# ---------------------------------------------------------

def logsumexp(t, w=None, axis=1):
    """
    t... tensor
    w... weight tensor
    """

    t_max = K.max(t, axis=axis, keepdims=True)

    if w is not None:
        tmp = w * K.exp(t - t_max)
    else:
        tmp = K.exp(t - t_max)

    out = K.sum(tmp, axis=axis)
    out = K.log(out)

    t_max = K.max(t, axis=axis)

    return out + t_max

# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------

class VisualisationCallback(keras.callbacks.Callback):
    """A callback for visualizing the progress in training."""

    def __init__(self, model, X_test, Y_test, epochs):

        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test
        self.epochs = epochs

        super(VisualisationCallback, self).__init__()

    def on_train_begin(self, logs={}):
        self.W_0 = self.model.get_weights()

    def on_epoch_begin(self, epoch, logs={}):
        self.plot_histogram(epoch)

    def on_train_end(self, logs={}):
        self.plot_histogram(epoch=self.epochs)
        images = []
        filenames = ["./.tmp%d.png" % epoch for epoch in np.arange(self.epochs + 1)]
        for filename in filenames:
            images.append(imageio.imread(filename))
            os.remove(filename)
        imageio.mimsave('./figures/retraining.gif', images, duration=.5)

    def plot_histogram(self, epoch):
        # get network weights
        W_T = self.model.get_weights()
        W_0 = self.W_0
        weights_0 = np.squeeze(special_flatten(W_0[:-3]))
        weights_T = np.squeeze(special_flatten(W_T[:-3]))
        # get means, variances and mixing proportions
        mu_T = np.concatenate([np.zeros(1), W_T[-3]]).flatten()
        prec_T = np.exp(W_T[-2])
        var_T = 1. / prec_T
        std_T = np.sqrt(var_T)
        pi_T = (np.exp(W_T[-1]))
        # plot histograms and GMM
        x0 = -1.2
        x1 = 1.2
        I = np.random.permutation(len(weights_0))
        f = sns.jointplot(weights_0[I], weights_T[I], size=8, kind="scatter", color="g", stat_func=None, edgecolor='w',
                          marker='o', joint_kws={"s": 8}, marginal_kws=dict(bins=1000), ratio=4)
        f.ax_joint.hlines(mu_T, x0, x1, lw=0.5)

        for k in range(len(mu_T)):
            if k == 0:
                f.ax_joint.fill_between(np.linspace(x0, x1, 10), mu_T[k] - 2 * std_T[k], mu_T[k] + 2 * std_T[k],
                                        color='blue', alpha=0.1)
            else:
                f.ax_joint.fill_between(np.linspace(x0, x1, 10), mu_T[k] - 2 * std_T[k], mu_T[k] + 2 * std_T[k],
                                        color='red', alpha=0.1)
        score = \
            self.model.evaluate({'input': self.X_test, }, {"error_loss": self.Y_test, "complexity_loss": self.Y_test, },
                                verbose=0)[3]
        sns.plt.title("Epoch: %d /%d\nTest accuracy: %.4f " % (epoch, self.epochs, score))
        f.ax_marg_y.set_xscale("log")
        f.set_axis_labels("Pretrained", "Retrained")
        f.ax_marg_x.set_xlim(-1, 1)
        f.ax_marg_y.set_ylim(-1, 1)
        display.clear_output()
        f.savefig("./.tmp%d.png" % epoch, bbox_inches='tight')
        plt.show()
