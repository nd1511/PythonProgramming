#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import tempfile
import numpy as np
import pytest
import renom.cuda
import renom.core
from renom.cuda import set_cuda_active, use_cuda, disable_cuda, use_device
from renom.core import to_value, Variable
from renom.operation import dot, sum, sqrt, square
from renom.config import precision
from renom.layers.function.gru import Gru
import renom as rm
import test_utility
from renom.layers.function.batch_normalize import BATCH_NORMALIZE_FEATUREMAP
import itertools

# if precision is not np.float32:
#    pytestmark = pytest.mark.skip()


def rand(shape):
    return np.array(np.random.rand(*shape), dtype=precision)


def randInt(shape):
    return np.array(np.random.randint(0, 2, shape), dtype=precision)


def arange(shape):
    return np.arange(np.prod(shape), dtype=precision).reshape(shape)


def close(a, b):
    assert np.allclose(to_value(a), to_value(b), atol=1e-4, rtol=1e-3)


def close_shape(a, b):
    assert a.shape == b.shape
    return close(a, b)


@test_utility.skipgpu
def test_gpu_node_neg():
    set_cuda_active(True)
    a = np.array(np.random.rand(10, )).astype(precision)

    g1 = Variable(a)
    g2 = -g1
    g2.to_cpu()

    set_cuda_active(False)
    close(g2, -g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((2, 3))],
    [rand((2, 3)), rand((3,))],
    [rand((3,)), rand((2, 3))],
    [rand((2, 3, 3)), rand((3, ))],
    [rand((2, 3, 3)), rand((3, 3, ))],
])
def test_gpu_node_add(a, b):
    with use_cuda():

        g1 = Variable(a)
        g2 = Variable(b)

        g3 = rm.sum(g1 + g2)
        g = g3.grad()

        g_g1 = g.get(g1)
        g_g2 = g.get(g2)
        g3.to_cpu()

    c3 = rm.sum(g1 + g2)
    c = c3.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)

    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((2, 3))],
    [rand((2, 3)), rand((3,))],
    [rand((3,)), rand((2, 3))],
    [rand((2, 3, 3)), rand((3, ))],
    [rand((2, 3, 3)), rand((3, 3, ))],
])
def test_gpu_node_mul(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sum(g1 * g2)
    g = g3.grad()

    g_g1 = g.get(g1)
    g_g2 = g.get(g2)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1 * g2)
    c = c3.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)

    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((2, 3))],
    [rand((2, 3)), rand((3,))],
    [rand((3,)), rand((2, 3))],
    [rand((2, 3, 3)), rand((3, ))],
    [rand((2, 3, 3)), rand((3, 3, ))],
])
def test_gpu_node_sub(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sum(g1 - g2)
    g = g3.grad()

    g_g1 = g.get(g1)
    g_g2 = g.get(g2)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1 - g2)
    c = c3.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)

    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((2, 3))],
    [rand((2, 3)), rand((3,))],
    [rand((3,)), rand((2, 3))],
    [rand((2, 3, 3)), rand((3, ))],
    [rand((2, 3, 3)), rand((3, 3, ))],
])
def test_gpu_node_div(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sum(g1 / g2)
    g = g3.grad()

    g_g1 = g.get(g1)
    g_g2 = g.get(g2)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1 / g2)
    c = c3.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)

    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((2, 3)),
    rand((2, 3)),
    rand((3,)),
    rand((2, 3, 3, 1)),
    rand((2, 3, 3, 1))
])
def test_gpu_node_abs(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(abs(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(abs(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((1, 3)), rand((3, 3))],
    [rand((2, 1)), rand((1, 3))],
    [rand((1, 1)), rand((1, 3))],
    [rand((1, 1)), rand((1, 1))],
])
def test_gpu_node_dot(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = dot(g1, g2)
    g4 = rm.sum(g3)
    g = g4.grad()
    g_g1 = g.get(g1)
    g_g2 = g.get(g2)
    g_g3 = g.get(g3)
    g3.to_cpu()
    g4.to_cpu()

    set_cuda_active(False)
    c3 = dot(g1, g2)
    c4 = rm.sum(c3)
    c = c4.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)
    c_c3 = c.get(c3)

    close(g3, c3)
    close(g4, c4)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_c3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((1, ))],
    [rand((3,)), rand((1, ))],
])
def test_gpu_node_pow(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sum(g1 ** g2)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1 ** g2)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)

# NEED to fix. Axis param


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    arange((2, 3)),
    arange((2, 1, 3)),
])
def test_gpu_node_sum(a):
    set_cuda_active(True)

    g1 = Variable(a)
    g3 = sum(g1)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(g1)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)

# NEED to fix. Axis param


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    arange((2, 3)),
    rand((2, 1)),
    # rand((2, 1, 2, 2))  # error
])
def test_gpu_node_sum_axis(a):
    set_cuda_active(True)

    g1 = Variable(a)
    g3 = sum(sum(g1, axis=0))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(sum(g1, axis=0))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((2, 3)),
    rand((2, 3, 3, 4)),
])
def test_gpu_node_sqrt(a):
    set_cuda_active(True)

    g1 = Variable(a)
    g3 = sum(sqrt(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(sqrt(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((2, 3)),
    rand((2, 3, 3, 4)),
])
def test_gpu_node_square(a):
    set_cuda_active(True)

    g1 = Variable(a)
    g3 = sum(square(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(square(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((1, 3)), randInt((1, 3))],
    [rand((2, 3)), randInt((2, 3))],
    [rand((1, 3, 3, 3)), randInt((1, 3, 3, 3))],
])
def test_gpu_node_softmax_cross_entropy(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.softmax_cross_entropy(g1, g2)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.softmax_cross_entropy(g1, g2)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((1, 3)), randInt((1, 3))],
    [rand((2, 3)), randInt((2, 3))],
])
def test_gpu_node_sigmoid_cross_entropy(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sigmoid_cross_entropy(g1, g2)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.sigmoid_cross_entropy(g1, g2)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((1, 3)), randInt((1, 3))],
    [rand((2, 3)), randInt((2, 3))],
    [rand((2, 3, 3)), randInt((2, 3, 3))],
])
def test_gpu_node_mean_squared_error(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.mean_squared_error(g1, g2)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.mean_squared_error(g1, g2)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_sigmoid(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.sigmoid(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.sigmoid(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_relu(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = sum(rm.relu(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(rm.relu(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_selu(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = sum(rm.selu(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(rm.selu(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_elu(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = sum(rm.elu(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(rm.elu(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_leaky_relu(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = sum(rm.leaky_relu(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(rm.leaky_relu(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_tanh(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.tanh(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.tanh(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_softmax(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.softmax(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.softmax(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_swish(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.swish(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.swish(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 4, 2)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_softsign(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.softsign(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.softsign(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@pytest.mark.skip()
@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_dropout(a):
    set_cuda_active(True)

    g1 = Variable(a)

    layer = rm.Dropout()

    np.random.seed(1)
    g3 = rm.sum(layer(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    np.random.seed(1)
    c3 = rm.sum(layer(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@pytest.mark.skip()
@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((4, 3, 3, 3)),
])
def test_gpu_node_spatial_dropout(a):
    with use_cuda():

        g1 = Variable(a)

        layer = rm.SpatialDropout()

        np.random.seed(1)
        g3 = rm.sum(layer(g1))
        g = g3.grad()
        g_g1 = g.get(g1)
        g3.to_cpu()

    np.random.seed(1)
    c3 = rm.sum(layer(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 1)),
    rand((1, 2)),
    rand((3, 3)),
    rand((3, 1)),
    rand((10, 9))
])
def test_gpu_dense(a):
    layer = rm.Dense(output_size=2)

    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(layer(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(layer(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 2)),
    rand((2, 2)),
])
def test_gpu_lstm(a):
    layer = rm.Lstm(output_size=2)

    def func(x):
        loss = 0
        for _ in range(5):
            loss += sum(layer(x))
        layer.truncate()
        return loss

    set_cuda_active(True)

    g1 = Variable(a)

    g3 = func(g1)
    g3.to_cpu()

    g = g3.grad()
    g_g1 = g.get(g1)
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = func(g1)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 2)),
    rand((2, 2)),
])
def test_gpu_gru(a):
    unit = Gru(output_size=2)

    def func(x):
        return sum(unit(x))

    set_cuda_active(True)

    g1 = Variable(a)

    g3 = func(g1)
    g3.to_cpu()

    g = g3.grad()
    g_g1 = g.get(g1)
    g_g1.to_cpu()

    set_cuda_active(False)
    unit.truncate()
    c3 = func(g1)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9)),
    rand((2, 3, 12, 9))
])
def test_gpu_node_convolution2d(a):
    with use_cuda():

        layer = rm.Conv2d(channel=32)
        layer.params["w"] = rm.Variable(np.random.rand(32, 3, 3, 3))
        layer.params["b"] = rm.Variable(np.random.rand(1, 32, 1, 1))

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g1 = g.get(layer.params["w"])
        g_g2 = g.get(layer.params["b"])
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad()
    c_g1 = c.get(layer.params["w"])
    c_g2 = c.get(layer.params["b"])
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9)),
    rand((2, 2, 2, 2,))
])
def test_gpu_node_convolutionnd(a):
    with use_cuda():

        layer = rm.ConvNd(channel=2, filter=1, stride=1, padding=0)
        #layer.params["w"] = rm.Variable(np.random.rand(32, 3, 3, 3))
        #layer.params["b"] = rm.Variable(np.random.rand(1, 32, 1, 1))

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g1 = g.get(layer.params["w"])
        g_g2 = g.get(layer.params["b"])
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad()
    c_g1 = c.get(layer.params["w"])
    c_g2 = c.get(layer.params["b"])
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9)),
    rand((2, 3, 12, 9))
])
def test_gpu_node_deconvolution2d(a):
    with use_cuda():

        layer = rm.Deconv2d(channel=32)
        layer.params["w"] = rm.Variable(np.random.rand(3, 32, 3, 3))
        layer.params["b"] = rm.Variable(np.random.rand(1, 32, 1, 1))

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g1 = g.get(layer.params["w"])
        g_g2 = g.get(layer.params["b"])
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad()
    c_g1 = c.get(layer.params["w"])
    c_g2 = c.get(layer.params["b"])
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((2, 3, 4, 5)),
])
def test_gpu_node_deconvolutionnd(a):
    a = np.ones_like(a)
    with use_cuda():

        layer = rm.DeconvNd(channel=2, filter=3, stride=1, padding=0)

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g1 = g.get(layer.params["w"])
        g_g2 = g.get(layer.params["b"])
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad()
    c_g1 = c.get(layer.params["w"])
    c_g2 = c.get(layer.params["b"])
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 16, 3, 3)),
    rand((1, 16, 9, 9)),
    rand((2, 16, 9, 9)),
    rand((2, 16, 12, 9))
])
def test_gpu_node_groupconvolution2d(a):
    with use_cuda():

        layer = rm.GroupConv2d(channel=32, groups=4)
        layer.params["w"] = rm.Variable(np.random.rand(32, 4, 3, 3))
        layer.params["b"] = rm.Variable(np.random.rand(1, 32, 1, 1))

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g1 = g.get(layer.params["w"])
        g_g2 = g.get(layer.params["b"])
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad()
    c_g1 = c.get(layer.params["w"])
    c_g2 = c.get(layer.params["b"])
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9)),
    rand((2, 3, 12, 9))
])
def test_gpu_node_max_pooling(a):
    with use_cuda():

        layer = rm.MaxPool2d()

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c3.grad()
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9, 4)),
    rand((2, 3, 12, 4))
])
def test_gpu_node_max_poolingNd(a):
    with use_cuda():

        layer = rm.MaxPoolNd()

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c3.grad()
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9)),
    rand((2, 3, 12, 9)),
    rand((2, 3, 12, 5))
])
def test_gpu_node_average_pooling(a):
    with use_cuda():

        layer = rm.AveragePool2d()

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c3.grad()
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a, mode", [
    [arange((4, 2)), 'activation'],
    [arange((4, 2, 3, 3)), 'feature'],
    [arange((4, 2, 3, 3)), 'activation'],
])
def test_batch_normalize(a, mode):
    layer = rm.Sequential([rm.BatchNormalize(momentum=0.5, mode=mode)])

    set_cuda_active(True)

    g1 = Variable(a)
    g2 = layer(g1)
    g3 = rm.sum(g2)
    g = g3.grad(detach_graph=False)
    g_g1 = g.get(g1)
    g_g2 = g.get(layer.l0.params["w"])
    g_g3 = g.get(layer.l0.params["b"])

    layer.set_models(inference=True)
    g4 = layer(g1)
    layer.set_models(inference=False)

    layer.save('temp.h5')
    layer.l0._mov_mean = 0
    layer.l0._mov_std = 0
    layer.load('temp.h5')
    layer.set_models(inference=True)
    g5 = layer(g1)
    layer.set_models(inference=False)

    g2.to_cpu()
    g3.to_cpu()
    g4.to_cpu()
    g_g1.to_cpu()
    g_g2.to_cpu()
    g_g3.to_cpu()

    set_cuda_active(False)
    layer.l0._mov_mean = 0
    layer.l0._mov_std = 0

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad(detach_graph=False)
    c_g1 = c.get(g1)
    c_g2 = c.get(layer.l0.params["w"])
    c_g3 = c.get(layer.l0.params["b"])

    layer.set_models(inference=True)
    c4 = layer(g1)
    layer.set_models(inference=False)

    layer.save('temp.h5')
    layer.l0._mov_mean = 0
    layer.l0._mov_std = 0
    layer.load('temp.h5')
    layer.set_models(inference=True)
    c5 = layer(g1)
    layer.set_models(inference=False)

    close(g2, c2)
    close(g3, c3)
    close(g4, c4)
    close(g5, c5)
    close(g4, g5)
    close(c4, c5)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)

    close(g2.attrs._m.new_array(), c2.attrs._m)
    close(g2.attrs._v.new_array(), c2.attrs._v)
    close(g2.attrs._mov_m.new_array(), c2.attrs._mov_m)
    close(g2.attrs._mov_v.new_array(), c2.attrs._mov_v)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((2, 3)),
])
def test_gpu_layer_normalize(a):
    set_cuda_active(True)

    g1 = Variable(a)

    layer = rm.LayerNormalize()

    g2 = layer(g1)
    g3 = rm.sum(g2)
    g = g3.grad(detach_graph=False)
    g_g1 = g.get(g1)
    g_g2 = g.get(layer.params["gain"])
    g_g3 = g.get(layer.params["bias"])

    set_cuda_active(False)

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad(detach_graph=False)
    c_c1 = c.get(g1)
    c_c2 = c.get(layer.params["gain"])
    c_c3 = c.get(layer.params["bias"])

    close(g2, c2)
    close(g3, c3)
    close(g_g1, c_c1)
    close(g_g2, c_c2)
    close(g_g3, c_c3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_reshape(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.reshape(g1, shape=(-1, 1)))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.reshape(g1, shape=(-1, 1)))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_flatten(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.flatten(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.flatten(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("node", [
    randInt((2, 3, 3, 3)),
])
def test_lrn(node):
    layer = rm.Lrn()

    with use_cuda():
        g1 = Variable(node)

        g3 = rm.sum(layer(g1))
        g = g3.grad()
        g_g1 = g.get(g1)

        g3.to_cpu()
        g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(layer(g1))

    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("node", [
    randInt((3, 2)),
    randInt((3, 2, 5, 1)),
])
def test_indexing(node):
    set_cuda_active(True)
    g1 = Variable(node)
    g3 = rm.sum(g1[1:2, -1])
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1[1:2, -1])
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("node", [
    randInt((3, 2)),
    randInt((3, 2, 5, 1)),
])
def test_where(node):
    #    set_cuda_active(is_active)

    with use_cuda():
        g1 = Variable(node)
        g3 = rm.sum(rm.where(g1 > 0.5, g1, 1))
        g = g3.grad()
        g_g1 = g.get(g1)
        g3.to_cpu()
        g_g1.to_cpu()

    with use_cuda():
        c3 = rm.sum(rm.where(g1 > 0.5, g1, 1))
        c = c3.grad()
        c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
def test_copy_from_cpu():
    src = Variable(rand((100,)))

    dest = Variable(rand((100,)))
    dest.copy_from(src)

    close(src, dest)


@test_utility.skipgpu
def test_copy_from_gpu():
    set_cuda_active(True)

    src = Variable(rand((100,)))
    src.to_gpu()

    dest = Variable(rand((100,)))
    dest.to_gpu()

    dest.copy_from(src)
    close(src, dest)

    close(src._gpu.new_array(), dest._gpu.new_array())


@test_utility.skipmultigpu
def test_copy_from_another_gpu():
    set_cuda_active(True)

    src = Variable(rand((100,)))
    src.to_gpu()

    with use_device(1):
        dest = Variable(rand((100,)))
        dest.to_gpu()

    dest.copy_from(src)
    close(src, dest)

    close(src._gpu.new_array(), dest._gpu.new_array())


@test_utility.skipgpu
@pytest.mark.parametrize("a, axis", [
    [rand((1, 2, 3)), None],
    [rand((1, 2, 3)), 0],
    [rand((1, 2, 3)), 1],
    [rand((1, 2, 3)), 2],
    [rand((1, 2, 3)), (0, 1)],
    [rand((1, 2, 3)), (0, 2)],
    [rand((1, 2, 3)), (0, 1, 2)],
])
def test_cusum(a, axis):
    with use_cuda():
        g = renom.core.GPUValue(a)

        ret = renom.cuda.cusum(g, axis, keepdims=False)
        close(ret.new_array(), np.sum(a, axis, keepdims=False))

        ret = renom.cuda.cusum(g, axis, keepdims=True)
        close(ret.new_array(), np.sum(a, axis, keepdims=True))


@test_utility.skipgpu
@pytest.mark.parametrize("a, axis", [
    [rand((1, 2, 3)), None],
    [rand((1, 2, 3)), 0],
    [rand((1, 2, 3)), 1],
    [rand((1, 2, 3)), 2],
    [rand((1, 2, 3)), (0, 1)],
    [rand((1, 2, 3)), (0, 2)],
    [rand((1, 2, 3)), (0, 1, 2)],
])
def test_cu_reduce_min(a, axis):
    with use_cuda():
        g = renom.core.GPUValue(a)

        ret = renom.cuda.cu_reduce_min(g, axis, keepdims=False)
        close_shape(ret.new_array(), np.min(a, axis, keepdims=False))

        ret = renom.cuda.cu_reduce_min(g, axis, keepdims=True)
        close_shape(ret.new_array(), np.min(a, axis, keepdims=True))


@test_utility.skipgpu
@pytest.mark.parametrize("a, axis", [
    [rand((1, 2, 3)), None],
    [rand((1, 2, 3)), 0],
    [rand((1, 2, 3)), 1],
    [rand((1, 2, 3)), 2],
])
def test_cu_reduce_arg_min(a, axis):
    with use_cuda():
        g = renom.core.GPUValue(a)

        ret = renom.cuda.cu_reduce_argmin(g, axis)
        close_shape(ret.new_array(), np.argmin(a, axis))


@test_utility.skipgpu
@pytest.mark.parametrize("a, axis", [
    [rand((1, 2, 3)), None],
    [rand((1, 2, 3)), 0],
    [rand((1, 2, 3)), 1],
    [rand((1, 2, 3)), 2],
    [rand((1, 2, 3)), (0, 1)],
    [rand((1, 2, 3)), (0, 2)],
    [rand((1, 2, 3)), (0, 1, 2)],
])
def test_cu_reduce_max(a, axis):
    with use_cuda():
        g = renom.core.GPUValue(a)

        ret = renom.cuda.cu_reduce_max(g, axis, keepdims=False)
        renom.cuda.cuDeviceSynchronize()
        close_shape(ret.new_array(), np.max(a, axis, keepdims=False))

        ret = renom.cuda.cu_reduce_max(g, axis, keepdims=True)
        close_shape(ret.new_array(), np.max(a, axis, keepdims=True))


@test_utility.skipgpu
@pytest.mark.parametrize("a, axis", [
    [rand((1, 2, 3)), None],
    [rand((1, 2, 3)), 0],
    [rand((1, 2, 3)), 1],
    [rand((1, 2, 3)), 2],
])
def test_cu_reduce_arg_max(a, axis):
    with use_cuda():
        g = renom.core.GPUValue(a)

        ret = renom.cuda.cu_reduce_argmax(g, axis)
        renom.cuda.cuDeviceSynchronize()
        close_shape(ret.new_array(), np.argmax(a, axis))


@test_utility.skipgpu
def test_transpose():
    with use_cuda():
        for n in range(0, 5):
            shape = [2 * (i + 1) for i in range(n)]
            a = np.arange(np.prod(shape)).reshape(shape).astype('float32')
            b = renom.core.GPUValue(a)
            for axis in itertools.permutations(range(len(shape))):
                aa = np.transpose(a, axis)
                bb = b.transpose(axis)

                assert np.allclose(aa, bb.new_array())


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((1,)), rand((1,))],
    [rand((2,)), rand((1,))],
    [rand((1, 3, 1)), rand((1,))],
    [rand((1, 3, 1)), rand((2,))],
    [rand((3, 1, 1)), rand((3, 3, 3))],
    [rand((2, 3, 1)), rand((3, 1,))],
    [rand((1, 2, 3, 1)), rand((3, 2,))],
    [rand((1, 2, 3, 1)), rand((2, 1, 3, 2,))],
    [rand((1, 2, 3, 1)), rand((2, 2, 3, 2,))],
    [rand((1, 1, 3)), rand((3, 3, 3,))],
])
def test_gpu_broadcast(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    assert np.allclose(a + b, (g1 + g2))


def comp_get(arr, f):
    with use_cuda():
        g = renom.core.GPUValue(arr)
        v1 = f(g)

    v2 = f(arr)
    assert np.allclose(v2, v1.new_array())


@test_utility.skipgpu
def test_getitem():
    s = np.arange(60).reshape(5, 3, 4)

    comp_get(s, lambda s: s[:])

    comp_get(s, lambda s: s[:, :, :])
    comp_get(s, lambda s: s[:, ..., :])
    comp_get(s, lambda s: s[..., :, :])

    comp_get(s, lambda s: s[0])

    comp_get(s, lambda s: s[0:1])
    comp_get(s, lambda s: s[2:3])
    comp_get(s, lambda s: s[2:3, 0, 1:3])
    comp_get(s, lambda s: s[::2])
    comp_get(s, lambda s: s[1:6:2, 0:1, 1:3])

    comp_get(s, lambda s: s[6:1:-2, 1:0, 1:3])
    comp_get(s, lambda s: s[6:1:-2, 1:0:-1, 1:3])
    comp_get(s, lambda s: s[6:1:-2, 1:0:-1, 3:1:-2])

    comp_get(s, lambda s: s[6:1:-2, 3:0:-1, 1:3])
    comp_get(s, lambda s: s[6:1:-2, 1:0:-1, 1:3])
    comp_get(s, lambda s: s[6:1:-2, 1:0:-1, 3:1:-2])
    comp_get(s, lambda s: s[1:1:-2, 1:0:-1, 3:1:-2])
    comp_get(s, lambda s: s[1:1:-2, 3:0:-2, 3:1:-2])
    comp_get(s, lambda s: s[1:1:-2, 3:0:-10, 3:1:-2])

    comp_get(s, lambda s: s[0, 3:0:-10, 3:1:-2])
    comp_get(s, lambda s: s[1:1:-2, 1, 3:1:-2])
    comp_get(s, lambda s: s[1:1:-2, 3:0:-10, 3])

    comp_get(s, lambda s: s[0, :, 2])
    comp_get(s, lambda s: s[0, :, None, 2])
    comp_get(s, lambda s: s[0, :, None, None, 2])
    comp_get(s, lambda s: s[None, 0, :, None, None, 2])

    comp_get(s, lambda s: s[1, 1, 1])


@test_utility.skipgpu
def test_getitem_advanced():

    s = np.arange(60).reshape(5, 3, 4)

    comp_get(s, lambda s: s[[0]])
    comp_get(s, lambda s: s[[0, 1]])
    comp_get(s, lambda s: s[[4, 2, 1]])
    comp_get(s, lambda s: s[[0, 1], :, [2, 1]])
    comp_get(s, lambda s: s[[0, 1], 0::2, [2, 1]])
    comp_get(s, lambda s: s[[0, 1], 0::-1, [2, 1]])
    comp_get(s, lambda s: s[[0, 1], 0::-2, [2, 1]])
    comp_get(s, lambda s: s[[[0, 1], [1, 0]], 0::-2, [[[2, 1], [1, 2]]]])
    comp_get(s, lambda s: s[[[0, 1]], 0::-2, [2, 1]])

    comp_get(s, lambda s: s[[0, 1], [2, 1]])
    comp_get(s, lambda s: s[[4, 2, 1]])

    comp_get(s, lambda s: s[:, [2, 1]])
    comp_get(s, lambda s: s[1:3, [1, 2], :])
    comp_get(s, lambda s: s[1:5:2, [1, 2]])
    comp_get(s, lambda s: s[:, [1, 0], :])

    comp_get(s, lambda s: s[[0], [2, 0]])
    comp_get(s, lambda s: s[[2], [1], [0, 1]])


@test_utility.skipgpu
def test_getitem_bool():

    s = np.arange(60).reshape(5, 3, 4)

    comp_get(s, lambda s: s[[True, False, True, False, True]])
    comp_get(s, lambda s: s[:, [True, False, True]])
    comp_get(s, lambda s: s[[True, False, True, False, False], :, [True, False, True, False]])

    ss = np.array([[1, 2], [3, 4]])

    comp_get(ss, lambda s: s[True])
    comp_get(ss, lambda s: s[False])
    comp_get(ss, lambda s: s[[True, False]])
    comp_get(ss, lambda s: s[[[True, False]]])
    comp_get(ss, lambda s: s[[True, False], [True, False]])
    comp_get(ss, lambda s: s[[[True, False], [True, False]]])
    comp_get(ss, lambda s: s[[[[True, False], [True, False]]]])

    s2 = np.array([[1, 1], [1, 1], [1, 1]])
    comp_get(s2, lambda s: s[np.array([[False, True], [True, False], [True, False]])])


@test_utility.skipgpu
def test_getitem_advanced2():
    s = np.arange(60).reshape(5, 3, 4)

    idx1 = np.array([[0, 1], [1, 0]])
    idx2 = np.array([[2, 1], [1, 2]])
    v1 = s[idx1, 0::-2, idx2]

    g = renom.core.GPUValue(s)
    v2 = g[idx1, 0::-2, idx2]

    np.allclose(v1, v2.new_array())


@test_utility.skipgpu
def test_getitem_advanced3():
    ss = np.array([[1, 2], [3, 4]])

    idx = np.array([[1, 0], [1, 0]])
    v1 = ss[idx]
    g = renom.core.GPUValue(ss)
    idx2 = renom.core.GPUValue(idx, dtype='int64')

    v2 = g[idx2]
    np.allclose(v1, v2.new_array())


@test_utility.skipgpu
def test_getitem_advanced4():
    ss = np.array([[1, 2], [3, 4]])
    g = renom.core.GPUValue(ss)

    idx = np.array([[True, False], [True, False]])
    v1 = ss[idx]
    v2 = g[idx]
    np.allclose(v1, v2.new_array())


@test_utility.skipgpu
def test_getitem_advanced5():
    ss = np.array([[1, 2], [3, 4]])
    g = renom.core.GPUValue(ss)

    idx = np.array([[True, False], [True, False]])
    v1 = ss[idx]

    idx2 = renom.core.GPUValue(idx, dtype='bool')
    v2 = g[idx2]
    np.allclose(v1, v2.new_array())


def comp_set(v):
    def deco(f):
        s = v.copy()
        g = renom.core.GPUValue(v)

        f(s)
        f(g)
        assert np.allclose(s, g.new_array())

    return deco


@test_utility.skipgpu
def test_setitem():
    s = np.arange(60).reshape(5, 3, 4)

    @comp_set(s)
    def test(s):
        s[:] = np.zeros(60).reshape(5, 3, 4)

    @comp_set(s)
    def test(s):
        s[:, :, ::2] = np.arange(30).reshape(5, 3, 2)

    @comp_set(s)
    def test(s):
        s[:, :, 1::2] = np.arange(30).reshape(5, 3, 2)

    @comp_set(s)
    def test(s):
        s[:, :, :2:2] = np.arange(15).reshape(5, 3, 1)
    #

    @comp_set(s)
    def test(s):
        s[:, :, ::-1] = np.arange(60).reshape(5, 3, 4)

    @comp_set(s)
    def test(s):
        s[:, :, ::-2] = np.arange(30).reshape(5, 3, 2)

    @comp_set(s)
    def test(s):
        s[1:3, :, ::-1] = np.arange(24).reshape(2, 3, 4)

    @comp_set(s)
    def test(s):
        s[3:1:-1, :, ::-1] = np.arange(24).reshape(2, 3, 4)

    @comp_set(s)
    def test(s):
        s[0, :, :] = np.arange(12).reshape(1, 3, 4)

    @comp_set(s)
    def test(s):
        s[0, 1, :] = np.arange(4).reshape(1, 1, 4)

    @comp_set(s)
    def test(s):
        s[0, 1, -1] = np.arange(1).reshape(1, 1, 1)

    @comp_set(s)
    def test(s):
        s[:, :, :] = np.arange(1).reshape(1)

    @comp_set(s)
    def test(s):
        s[[0, 1], :, [1, 0]] = np.array([999]).reshape(1, 1, 1, 1, 1)

    @comp_set(s)
    def test(s):
        s[[0, 1], :, [1, 0]] = np.array([999] * 3)

    @comp_set(s)
    def test(s):
        s[:] = np.array([999] * 4)

    @comp_set(s)
    def test(s):
        s[:] = np.array([[999], [1000], [1001]])

    @comp_set(s)
    def test(s):
        s[:] = np.array([
            [[999]],
            [[1000]],
            [[1001]],
            [[1002]],
            [[1003]],
        ])

    @comp_set(s)
    def test(s):
        s[:] = np.array([
            [[999, 1, 2, 3]],
            [[1000, 2, 3, 4]],
            [[1001, 4, 5, 6]],
            [[1002, 6, 7, 8]],
            [[1003, 8, 9, 9]],
        ])

    @comp_set(s)
    def test(s):
        s[:, :, [True, False, False, True]] = np.array([1, 2])

    @comp_set(s)
    def test(s):
        s[:, :, [True, False, False, True]] = np.array([1])

    @comp_set(s)
    def test(s):
        s[:] = np.array(1)

    @comp_set(s)
    def test(s):
        s[[True, False, True, False, True]] = np.arange(36).reshape(3, 3, 4)

    @comp_set(s)
    def test(s):
        s[[True, False, True, False, True]] = np.arange(4)

    @comp_set(s)
    def test(s):
        s[[True, False, True, False, True]] = np.arange(1)

    @comp_set(s)
    def test(s):
        s[:, [True, False, True]] = np.arange(40).reshape(5, 2, 4)

    @comp_set(s)
    def test(s):
        s[:, [True, False, True]] = np.arange(4)

    @comp_set(s)
    def test(s):
        s[:, [True, False, True]] = np.arange(1)

    @comp_set(s)
    def test(s):
        s[[True, False, True, False, False], :, [
            True, False, True, False]] = np.arange(6).reshape(2, 3)

    ss = np.array([[1, 2], [3, 4]])

    @comp_set(ss)
    def test(s):
        s[True] = np.arange(4).reshape(1, 2, 2)

    @comp_set(ss)
    def test(s):
        s[True] = np.arange(1)

    @comp_set(ss)
    def test(s):
        s[False] = np.arange(4).reshape(1, 2, 2)

    @comp_set(ss)
    def test(s):
        s[[True, False]] = np.arange(2).reshape(1, 2)

    @comp_set(ss)
    def test(s):
        s[[[True, False]]] = np.arange(2).reshape(1, 2)

    @comp_set(ss)
    def test(s):
        s[[[True, False]]] = np.arange(2).reshape(1, 2)

    @comp_set(ss)
    def test(s):
        s[[True, False], [True, False]] = np.arange(1)

    @comp_set(ss)
    def test(s):
        s[[[True, False], [True, False]]] = np.arange(1)

    @comp_set(ss)
    def test(s):
        s[[[[True, False], [True, False]]]] = np.arange(2)

    @comp_set(np.array([[1, 1], [1, 1], [1, 1]]))
    def test(s):
        s[[[[False, True], [True, False], [True, False]]]] = np.arange(3)


def comp_splitted(nd, gpu):
    for n, g in zip(nd, gpu):
        g = g.new_array()
        close(n, g)


@test_utility.skipgpu
def test_split():
    v = np.arange(0, 24).reshape(2, 3, 4)
    g = renom.core.GPUValue(v)

    comp_splitted(np.split(v, 2, 0), g.split(2, 0))
    comp_splitted(np.split(v, 2, 2), g.split(2, 2))
    comp_splitted(np.split(v, [1, 1], 2), g.split([1, 1], 2))
    comp_splitted(np.split(v, [1, 2], 2), g.split([1, 2], 2))
    comp_splitted(np.split(v, [1, 3], 2), g.split([1, 3], 2))
    comp_splitted(np.split(v, [1, 4], 2), g.split([1, 4], 2))


@test_utility.skipgpu
def test_hplit():
    v = np.arange(0, 24).reshape(2, 3, 4)
    g = renom.core.GPUValue(v)

    comp_splitted(np.hsplit(v, [0, 2]), g.hsplit([0, 2]))


@test_utility.skipgpu
def test_split_err():
    v = np.arange(0, 24).reshape(2, 3, 4)
    g = renom.core.GPUValue(v)

    with pytest.raises(ValueError):
        g.split(2, 1)

    with pytest.raises(IndexError):
        g.split(2, 3)
