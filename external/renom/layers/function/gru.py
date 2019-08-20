#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.layers.activation.sigmoid import sigmoid
from renom.layers.activation.tanh import tanh
from renom.core import Node, Variable, GetItem
from renom import precision
from renom.operation import dot, sum, concat
from renom.utility.initializer import GlorotNormal
from .parameterized import Parametrized
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_diff(x):
    return (1.0 - tanh(x) ** 2)


class gru(Node):
    '''
    @ parameters
    cls: Self variable for Python
    x: input value to the node
    pz: The previously calculated value within the same model
    w: the weights to be multiplied with the input
    u: the weights to be multiplied with the previous input
    b: the biases to be added
    '''
    def __new__(cls, x, pz, w, u, b):
        return cls.calc_value(x, pz, w, u, b)

    @classmethod
    def _oper_cpu(cls, x, pz, w, u, b):
        # Initialize Variables
        m = w.shape[1] // 3
        w_z, w_r, w_h = np.split(w, [m, m * 2, ], axis=1)
        u_z, u_r, u_h = np.split(u, [m, m * 2], axis=1)
        hminus = Variable(np.zeros((x.shape[0], w.shape[1] // 3),
                                   dtype=precision)) if pz is None else pz

        # Perform Forward Calcuations
        if b is None:
            A = dot(x, w_z) + hminus * u_z
            B = dot(x, w_r) + u_r * hminus
            C = dot(x, w_h) + sigmoid(B) * u_h * hminus
        else:
            b_z, b_r, b_h = np.split(b, [m, m * 2], axis=1)
            A = dot(x, w_z) + hminus * u_z + b_z
            B = dot(x, w_r) + u_r * hminus + b_r
            C = dot(x, w_h) + sigmoid(B) * u_h * hminus + b_h

        h = sigmoid(A) + tanh(C)

        # Store Variables for Graph
        ret = cls._create_node(h)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._w_z = w_z
        ret.attrs._w_r = w_r
        ret.attrs._w_h = w_h
        ret.attrs._b = b
        ret.attrs._b_z = b_z
        ret.attrs._b_r = b_r
        ret.attrs._b_h = b_h
        ret.attrs._u = u
        ret.attrs._u_z = u_z
        ret.attrs._u_h = u_h
        ret.attrs._u_r = u_r
        ret.attrs._pz = hminus
        ret.attrs._A = A
        ret.attrs._B = B
        ret.attrs._C = C

        return ret

    @classmethod
    def _oper_gpu(cls, x, pz, w, u, b):
        # Initialize Variables
        m = w.shape[1] // 3
        hminus = Variable(np.zeros((x.shape[0], m), dtype=precision)) if pz is None else pz
        get_gpu(hminus)
        # Perform Forward Calcuations
        input = dot(get_gpu(x), get_gpu(w)) + get_gpu(b)
        ABC = get_gpu(input).empty_like_me()
        h = get_gpu(hminus).empty_like_me()
        cu.cugru_forward(get_gpu(input), get_gpu(hminus), get_gpu(u), get_gpu(ABC), get_gpu(h))

        # Store Variables for Graph
        ret = cls._create_node(h)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._u = u
        ret.attrs._pz = hminus
        ret.attrs._ABC = ABC

        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        x = self.attrs._x
        w_z = self.attrs._w_z
        w_r = self.attrs._w_r
        w_h = self.attrs._w_h
        A = self.attrs._A
        B = self.attrs._B
        C = self.attrs._C
        u_z = self.attrs._u_z
        u_h = self.attrs._u_h
        u_r = self.attrs._u_r
        hminus = self.attrs._pz
        y = dy

        dA = sigmoid_diff(A)
        dB = sigmoid_diff(B)
        dC = tanh_diff(C)

        # Calculate dx
        dx_z = dot(y * dA, w_z.T)
        dx_r = dot(y * dB * dC * u_h * hminus, w_r.T)
        dx_h = dot(y * dC, w_h.T)
        dx = dx_z + dx_r + dx_h

        # Calculate dw
        dw_z = dot(x.T, y * dA)
        dw_r = dot(x.T, y * dB * dC * u_h * hminus)
        dw_h = dot(x.T, y * dC)
        dw = np.concatenate([dw_z, dw_r, dw_h], axis=1)

        # Calculate db
        db_z = np.sum(y * dA, axis=0, keepdims=True)
        db_r = np.sum(y * dB * dC * u_h * hminus, axis=0, keepdims=True)
        db_h = np.sum(y * dC, axis=0, keepdims=True)
        db = np.concatenate([db_z, db_r, db_h], axis=1)

        du_z = np.sum(dA * hminus * y, axis=0, keepdims=True)
        du_r = np.sum(y * dC * dB * u_h * hminus * hminus, axis=0, keepdims=True)
        du_h = np.sum(sigmoid(B) * dC * y * hminus, axis=0, keepdims=True)
        du = np.concatenate([du_z, du_r, du_h], axis=1)

        pz_z = y * dA * u_z
        pz_r = y * dC * dB * u_h * hminus * u_r
        pz_h = y * dC * sigmoid(B) * u_h

        dpz = pz_z + pz_r + pz_h

        self.attrs._x._update_diff(context, dx)
        self.attrs._w._update_diff(context, dw)
        self.attrs._b._update_diff(context, db)
        self.attrs._u._update_diff(context, du)
        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dpz)

    def prn(self, v, name='Node'):
        h = self._create_node(v)
        h.to_cpu()
        print('{:10}= {}'.format(name, h))

    def _backward_gpu(self, context, dy, **kwargs):
        x = self.attrs._x
        w = self.attrs._w
        b = self.attrs._b
        u = self.attrs._u
        hminus = self.attrs._pz
        ABC = self.attrs._ABC

        dx = get_gpu(x).empty_like_me()
        db = get_gpu(b).empty_like_me()
        yconc = get_gpu(ABC).empty_like_me()
        du = get_gpu(u).empty_like_me()
        dpz = get_gpu(hminus).empty_like_me()
        dxx = get_gpu(x).empty_like_me()

        cu.cugru_backward(get_gpu(ABC), get_gpu(dy), yconc, get_gpu(u),
                          get_gpu(hminus), db, du, dpz, dxx)
        # Calculate dx

        dx = get_gpu(dot(yconc, w.T))

        xconc = get_gpu(x.T)

        dw = dot(get_gpu(xconc), get_gpu(yconc))

        self.attrs._x._update_diff(context, dx)
        self.attrs._w._update_diff(context, dw)
        self.attrs._b._update_diff(context, db)
        self.attrs._u._update_diff(context, du)
        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dpz)


class Gru(Parametrized):
    '''
    Gated Recurrent Unit

    An LSTM-like RNN unit, which simplifies the LSTM unit by not including a memory core.
    This simplifies learning of the unit and reduces computational complexity, as the GRU only
    performs requires 3 input gates, compared to the 4 required by the LSTM.

    Args:
        output_size (int): Output unit size.
        input_size (int): Input unit size.
        ignore_bias (bool): If True is given, bias will not be added.
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> n, d, t = (2, 3, 4)
        >>> x = rm.Variable(np.random.rand(n,d))
        >>> layer = rm.Gru(2)
        >>> z = 0
        >>> for i in range(t):
        ...     z += rm.sum(layer(x))
        ...
        >>> grad = z.grad()
        >>> grad.get(x)
        Add([[-8.89559174, -0.58861321, -4.67931843],
        [-7.27466679, -0.45286781, -3.81758523]], dtype=float32)
        >>> layer.truncate()

    https://arxiv.org/pdf/1409.1259.pdf

    '''

    def __init__(self, output_size, input_size=None, ignore_bias=False, initializer=GlorotNormal(),
                 weight_decay=0):
        self._size_o = output_size
        self._initializer = initializer
        self._ignore_bias = ignore_bias
        self._weight_decay = weight_decay
        super(Gru, self).__init__(input_size)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        bias = np.ones((1, size_o * 3), dtype=precision)
        # At this point, all connected units in the same layer will use the SAME weights
        self.params = {
            "w": Variable(self._initializer((size_i, size_o * 3)), auto_update=True, weight_decay=self._weight_decay),
            "u": Variable(self._initializer((1, size_o * 3)), auto_update=True, weight_decay=self._weight_decay),
        }
        if not self._ignore_bias:
            self.params["b"] = Variable(bias, auto_update=True)

    def forward(self, x):
        ret = gru(x, getattr(self, "_z", None),
                  self.params.w,
                  self.params.u,
                  self.params.get("b", None))
        self._z = ret
        return ret

    def truncate(self):
        """Truncates temporal connection."""
        self._z = None
