#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Methods for gerneral purpose

Karen Ullrich, Sep 2016
"""

import numpy as np
from scipy.misc import logsumexp

import matplotlib.pyplot as plt
import seaborn as sns
# ---------------------------------------------------------
# RESHAPING LISTS FILLED WITH ARRAYS
# ---------------------------------------------------------

def special_flatten(arraylist):
    """Flattens the output of model.get_weights()"""
    out = np.concatenate([array.flatten() for array in arraylist])
    return out.reshape((len(out), 1))


def reshape_like(in_array, shaped_array):
    "Inverts special_flatten"
    flattened_array = list(in_array)
    out = np.copy(shaped_array)
    for i, array in enumerate(shaped_array):
        num_samples = array.size
        dummy = flattened_array[:num_samples]
        del flattened_array[:num_samples]
        out[i] = np.asarray(dummy).reshape(array.shape)
    return out


# ---------------------------------------------------------
# DISCRETESIZE
# ---------------------------------------------------------

def merger(inputs):
    """Comparing and merging components."""
    for _ in xrange(3):
        lists = []
        for inpud in inputs:
            for i in inpud:
                tmp = 1
                for l in lists:
                    if i in l:
                        for j in inpud:
                            l.append(j)
                        tmp = 0
                if tmp is 1:
                    lists.append(list(inpud))
        lists = [np.unique(l) for l in lists]
        inputs = lists
    return lists


def KL(means, logprecisions):
    """Compute the KL-divergence between 2 Gaussian Components."""
    precisions = np.exp(logprecisions)
    return 0.5 * (logprecisions[0] - logprecisions[1]) + precisions[1] / 2. * (
    1. / precisions[0] + (means[0] - means[1]) ** 2) - 0.5


def compute_responsibilies(xs, mus, logprecisions, pis):
    "Computing the unnormalized responsibilities."
    xs = xs.flatten()
    K = len(pis)
    W = len(xs)
    responsibilies = np.zeros((K, len(xs)))
    for k in xrange(K):
        # Not normalized!!!
        responsibilies[k] = pis[k] * np.exp(0.5 * logprecisions[k]) * np.exp(
            - np.exp(logprecisions[k]) / 2 * (xs - mus[k]) ** 2)
    return np.argmax(responsibilies, axis=0)


def discretesize(W, pi_zero=0.999):
    # flattening hte weights
    weights = special_flatten(W[:-3])

    means = np.concatenate([np.zeros(1), W[-3]])
    logprecisions = W[-2]
    logpis = np.concatenate([np.log(pi_zero) * np.ones(1), W[-1]])

    # classes K
    J = len(logprecisions)
    # compute KL-divergence
    K = np.zeros((J, J))
    L = np.zeros((J, J))

    for i, (m1, pr1, pi1) in enumerate(zip(means, logprecisions, logpis)):
        for j, (m2, pr2, pi2) in enumerate(zip(means, logprecisions, logpis)):
            K[i, j] = KL([m1, m2], [pr1, pr2])
            L[i, j] = np.exp(pi1) * (pi1 - pi2 + K[i, j])

    # merge
    idx, idy = np.where(K < 1e-10)
    lists = merger(np.asarray(zip(idx, idy)))
    # compute merged components
    # print lists
    new_means, new_logprecisions, new_logpis = [], [], []

    for l in lists:
        new_logpis.append(logsumexp(logpis[l]))
        new_means.append(
            np.sum(means[l] * np.exp(logpis[l] - np.min(logpis[l]))) / np.sum(np.exp(logpis[l] - np.min(logpis[l]))))
        new_logprecisions.append(np.log(
            np.sum(np.exp(logprecisions[l]) * np.exp(logpis[l] - np.min(logpis[l]))) / np.sum(
                np.exp(logpis[l] - np.min(logpis[l])))))

    new_means[np.argmin(np.abs(new_means))] = 0.0

    # compute responsibilities
    argmax_responsibilities = compute_responsibilies(weights, new_means, new_logprecisions, np.exp(new_logpis))
    out = [new_means[i] for i in argmax_responsibilities]

    out = reshape_like(out, shaped_array=W[:-3])
    return out


def save_histogram(W_T,save, upper_bound=200):
        w = np.squeeze(special_flatten(W_T[:-3]))
        plt.figure(figsize=(10, 7))
        sns.set(color_codes=True)
        plt.xlim(-1,1)
        plt.ylim(0,upper_bound)
        sns.distplot(w, kde=False, color="g",bins=200,norm_hist=True)
        plt.savefig("./"+save+".png", bbox_inches='tight')
        plt.close()


        plt.figure(figsize=(10, 7))
	plt.yscale("log")
        sns.set(color_codes=True)
        plt.xlim(-1,1)
        plt.ylim(0.001,upper_bound*5)
        sns.distplot(w, kde=False, color="g",bins=200,norm_hist=True)
        plt.savefig("./"+save+"_log.png", bbox_inches='tight')
        plt.close()
