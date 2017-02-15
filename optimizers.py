"""
A modified copy of Keras Adam Optimizer.

Author: Karen Ullrich, Sep 2016

"""

from __future__ import print_function
import numpy as np

from keras import backend as K
from keras.utils.generic_utils import get_from_module

from keras.optimizers import Optimizer


class Adam(Optimizer):
    """Adam optimizer.
    An extended Version. parameters that have been named can be trained with
    different hyperparams.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self,
                 lr=[0.001],
                 beta_1=None,
                 beta_2=None,
                 epsilon=1e-8,
                 decay=None,
                 param_types_dict=[],
                 **kwargs):

        super(Adam, self).__init__(**kwargs)

        if lr is None:
            lr = [0.001]
        self.__dict__.update(locals())

        self.iterations = K.variable(0)
        # init params if not set
        l = len(lr)
        if beta_1 is None:
            beta_1 = list(np.tile([0.9], l))
        if beta_2 is None:
            beta_2 = list(np.tile([0.999], l))
        if decay is None:
            decay = list(np.tile([0.], l))
        # add a tag for non-tagged variables
        self.param_types_dict = ['other'] + param_types_dict

        self.lr = {}
        self.beta_1, self.beta_2 = {}, {}
        self.decay, self.inital_decay = {}, {}

        for param_type in self.param_types_dict:
            self.lr[param_type] = K.variable(lr.pop(0))
            self.beta_1[param_type] = K.variable(beta_1.pop(0))
            self.beta_2[param_type] = K.variable(beta_2.pop(0))
            tmp = decay.pop(0)
            self.decay[param_type] = K.variable(tmp)
            self.inital_decay[param_type] = tmp

        self.epsilon = epsilon

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = self.iterations + 1

        lr_t = {}
        for param_type in self.param_types_dict:
            lr = self.lr[param_type]
            if self.inital_decay[param_type] > 0:
                lr *= (1. / (1. + self.decay[param_type] * self.iterations[param_type]))
            lr_t[param_type] = lr * K.sqrt(1. - K.pow(self.beta_2[param_type], t)) / (
                1. - K.pow(self.beta_1[param_type], t))

        shapes = [K.get_variable_shape(p) for p in params]
        # add param type here
        param_types = []
        for param in params:
            tmp = None
            for param_type in self.param_types_dict:
                if param_type in param.name:
                    tmp = param_type
            if tmp is None:
                tmp = 'other'
            param_types.append(tmp)

        if len(param_types) != len(params):
            print('Something went wrong with the naming of variables.')

        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, param_type, g, m, v in zip(params, param_types, grads, ms, vs):
            m_t = (self.beta_1[param_type] * m) + (1. - self.beta_1[param_type]) * g
            v_t = (self.beta_2[param_type] * v) + (1. - self.beta_2[param_type]) * K.square(g)
            p_t = p - lr_t[param_type] * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))

        return self.updates

    @property
    def get_config(self):

        lr = {}
        beta_1, beta_2 = {}, {}
        decay, inital_decay = {}, {}

        for param_type in self.param_types_dict:
            lr[param_type] = float(K.get_value(self.lr[param_type]))
            beta_1[param_type] = float(K.get_value(self.beta_1[param_type]))
            beta_2[param_type] = float(K.get_value(self.beta_2[param_type]))
            decay[param_type] = float(K.get_value(self.decay[param_type]))
            inital_decay[param_type] = float(K.get_value(self.inital_decay[param_type]))

        config = {'lr': lr,
                  'beta_1': beta_1,
                  'beta_2': beta_2,
                  'epsilon': self.epsilon}

        base_config = super(Adam, self).get_config

        return dict(list(base_config.items()) + list(config.items()))


# aliases
adam = Adam


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer',
                           instantiate=True, kwargs=kwargs)
