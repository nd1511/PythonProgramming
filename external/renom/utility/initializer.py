#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from builtins import object
import numpy as np
from renom.config import precision


class Initializer(object):
    """Base class of initializer.

    When the initialization of parameterized layer class,
    dense, conv2d, lstm ... , you can select the initialization method
    changing the initializer class as following example.

    Example:
        >>> import renom as rm
        >>> from renom.utility.initializer import GlorotUniform
        >>>
        >>> layer = rm.Dense(output_size=2, input_size=2, initializer=GlorotUniform())
        >>> print(layer.params.w)
        [[-0.55490332 -0.14323548]
         [ 0.00059367 -0.28777076]]

    """

    def __call__(self, shape):
        raise NotImplementedError


class GlorotUniform(Initializer):

    '''Glorot uniform initializer [1]_ initializes parameters sampled by
    following uniform distribution "U(max, min)".

    .. math::

        &U(max, min) \\\\
        &max = sqrt(6/(input\_size + output\_size)) \\\\
        &min = -sqrt(6/(input\_size + output\_size))

    '''

    def __init__(self):
        super(GlorotUniform, self).__init__()

    def __call__(self, shape):
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:
            size = np.prod(shape[2:])
            fan_in = shape[0] * size
            fan_out = shape[1] * size
        lim = np.sqrt(6 / (fan_in + fan_out))
        return (np.random.rand(*shape) * 2 * lim - lim).astype(precision)


class GlorotNormal(Initializer):

    '''Glorot normal initializer [1]_ initializes parameters sampled by
    following normal distribution "N(0, std)".

    .. math::

        &N(0, std) \\\\
        &std = sqrt(2/(input\_size + output\_size)) \\\\

    .. [1] Xavier Glorot, Yoshua Bengio.
        Understanding the difficulty of training deep feedforward neural networks.
    '''

    def __init__(self):
        super(GlorotNormal, self).__init__()

    def __call__(self, shape):
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:
            size = np.prod(shape[2:])
            fan_in = shape[0] * size
            fan_out = shape[1] * size
        std = np.sqrt(2 / (fan_in + fan_out))
        return (np.random.randn(*shape) * std).astype(precision)


class HeNormal(Initializer):

    '''He normal initializer.
   Initializes parameters according to [1]

    .. math::

        &N(0, std) \\\\
        &std = sqrt(2/(input\_size)) \\\\

    .. [1] https://arxiv.org/abs/1502.01852
       Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

    '''

    def __init__(self):
        super(HeNormal, self).__init__()

    def __call__(self, shape):
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) == 4:
            size = np.prod(shape[2:])
            fan_in = shape[1] * size
        std = np.sqrt(2 / fan_in)
        return (np.random.randn(*shape) * std).astype(precision)


class HeUniform(Initializer):

    '''He uniform initializer.
   Initializes parameters according to [1]

    .. math::


        &U(max, min) \\\\
        &max = sqrt(6/(input\_size)) \\\\
        &min = -sqrt(6/(input\_size))

    .. [1] https://arxiv.org/abs/1502.01852
       Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

    '''

    def __init__(self):
        super(HeUniform, self).__init__()

    def __call__(self, shape):
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) == 4:
            size = np.prod(shape[2:])
            fan_in = shape[1] * size
        lim = np.sqrt(6 / (fan_in))
        return (np.random.rand(*shape) * 2 * lim - lim).astype(precision)


class Gaussian(Initializer):

    '''Gaussian initializer.
    Initialize parameters using samples drawn from :math:`N(mean, std)`

    Args:
        mean (float): Mean value of normal distribution.
        std (float): Standard deviation value of normal distribution.

    '''

    def __init__(self, mean=0.0, std=0.1):
        super(Gaussian, self).__init__()
        self._mean = mean
        self._std = std

    def __call__(self, shape):
        return (np.random.randn(*shape) * self._std + self._mean).astype(precision)


class Uniform(Initializer):

    '''Uniform initializer.
    Initialize parameters using samples drawn from :math:`U(min, max)`

    Args:
        min (float): Minimum limit of uniform distribution.
        max (float): Maximum limit of uniform distribution.

    '''

    def __init__(self, min=-1.0, max=1.0):
        super(Uniform, self).__init__()
        self._min = min
        self._max = max

    def __call__(self, shape):
        shape[1:]
        delt = self._max - self._min
        return (np.random.rand(*shape) * delt + self._min).astype(precision)
