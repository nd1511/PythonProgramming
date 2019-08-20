#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division
import numpy as np
import onnx
import renom as rm
import renom.utility.onnx
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE


def _run_onnx(tmpdir, model, input):
    f = tmpdir / "test1.onnx"
    renom.utility.onnx.export_onnx("test1", model, input, str(f))

    return onnx.load(str(f))


def get_shape(typedef):
    shape = []
    for dim in typedef.type.tensor_type.shape.dim:
        shape.append(dim.dim_value)
    return tuple(shape)


def load_initializer(initializer):
    ret = {}
    for ini in initializer:
        shape = ini.dims
        dtype = TENSOR_TYPE_TO_NP_TYPE[ini.data_type]
        v = np.frombuffer(ini.raw_data, dtype=dtype).reshape(shape)
        ret[ini.name] = v
    return ret


def _test_initializer(d, name, obj):
    assert np.allclose(d[name], obj)


def test_dense(tmpdir):
    class Model(rm.Model):
        def __init__(self):
            super(Model, self).__init__()
            self.layer1 = rm.Dense(output_size=10)

        def forward(self, x):
            x = self.layer1(x)
            return x

    model = Model()
    input = renom.Variable(np.random.random((100, 5)))

    m = _run_onnx(tmpdir, model, input)

    assert m.graph.node[0].op_type == 'Gemm'
    id_input, w, b = m.graph.node[0].input

    inis = load_initializer(m.graph.initializer)

    # check input
    assert 'input' == id_input
    assert get_shape(m.graph.input[0]) == input.shape

    # check w
    assert get_shape(m.graph.input[1]) == model.layer1.params.w.shape
    _test_initializer(inis, w, model.layer1.params.w)

    # check b
    assert get_shape(m.graph.input[2]) == model.layer1.params.b.shape
    _test_initializer(inis, b, model.layer1.params.b)

    # check output
    assert get_shape(m.graph.output[0]) == (100, 10)


def test_conv2d(tmpdir):
    model = rm.Sequential([
        rm.Conv2d(channel=32, filter=3, padding=1),
    ])

    input = renom.Variable(np.random.random((10, 10, 10, 10)))
    m = _run_onnx(tmpdir, model, input)

    assert m.graph.node[0].op_type == 'Conv'

    attrs = dict((a.name, a) for a in m.graph.node[0].attribute)
    assert attrs['pads'].ints == [1, 1]
    assert attrs['dilations'].ints == [1, 1]
    assert attrs['kernel_shape'].ints == [3, 3]
    assert attrs['strides'].ints == [1, 1]

    id_input, w, b = m.graph.node[0].input

    # check input
    assert 'input' == id_input
    assert get_shape(m.graph.input[0]) == input.shape

    conv2d = model._layers[0]
    inis = load_initializer(m.graph.initializer)

    # check w
    assert get_shape(m.graph.input[1]) == conv2d.params.w.shape
    _test_initializer(inis, w, conv2d.params.w)

    # check b
    assert get_shape(m.graph.input[2]) == (np.prod(conv2d.params.b.shape),)
    _test_initializer(inis, b, conv2d.params.b)

    # check output
    assert get_shape(m.graph.output[0]) == (10, 32, 10, 10)


def _test_unary(tmpdir, f, name):
    class Model(rm.Model):
        def forward(self, x):
            return f(x)

    model = Model()
    input = np.random.random((2, 2))
    m = _run_onnx(tmpdir, model, input)

    assert m.graph.node[0].op_type == name

    # check input
    id_input, = m.graph.node[0].input
    assert 'input' == id_input
    assert get_shape(m.graph.input[0]) == input.shape

    # check output
    assert get_shape(m.graph.output[0]) == (2, 2)


def test_unary(tmpdir):
    _test_unary(tmpdir, (lambda x: -x), 'Neg')
    _test_unary(tmpdir, abs, 'Abs')


def _test_binop(tmpdir, f, name):
    arg = rm.Variable(np.random.random((2, 2)))

    class Model(rm.Model):
        def forward(self, x):
            return f(x, arg)

    model = Model()
    input = np.random.random((2, 2))
    m = _run_onnx(tmpdir, model, input)

    # check input
    assert m.graph.node[0].op_type == name
    input, rhs = m.graph.node[0].input

    # check lhs
    inis = load_initializer(m.graph.initializer)
    _test_initializer(inis, rhs, arg)

    # lhs should never has initializer
    assert input not in inis


def test_binops(tmpdir):
    _test_binop(tmpdir, (lambda x, y: x + y), 'Add')
    _test_binop(tmpdir, (lambda x, y: x - y), 'Sub')
    _test_binop(tmpdir, (lambda x, y: x * y), 'Mul')
    _test_binop(tmpdir, (lambda x, y: x / y), 'Div')


def test_relu(tmpdir):
    model = rm.Sequential([
        rm.Relu()
    ])

    input = renom.Variable(np.random.random((10, 10, 10, 10)))
    m = _run_onnx(tmpdir, model, input)

    assert m.graph.node[0].op_type == 'Relu'

    # check input
    id_input, = m.graph.node[0].input
    assert renom.utility.onnx.OBJNAMES[id(input)] == id_input
    assert id_input == m.graph.input[0].name
    assert get_shape(m.graph.input[0]) == input.shape

    # check output
    id_output, = m.graph.node[0].output
    assert get_shape(m.graph.output[0]) == input.shape


def test_max_pool2d(tmpdir):
    model = rm.Sequential([
        rm.MaxPool2d(filter=2, stride=2),
    ])

    input = renom.Variable(np.random.random((10, 10, 10, 10)))
    m = _run_onnx(tmpdir, model, input)

    # check input
    id_input, = m.graph.node[0].input
    assert 'input' == id_input
    assert get_shape(m.graph.input[0]) == input.shape

    # check output
    assert get_shape(m.graph.output[0]) == (10, 10, 5, 5)

    # check attrs
    attrs = dict((a.name, a) for a in m.graph.node[0].attribute)
    assert attrs['pads'].ints == [0, 0]
    assert attrs['kernel_shape'].ints == [2, 2]
    assert attrs['strides'].ints == [2, 2]


def test_dropout(tmpdir):
    model = rm.Sequential([
        rm.Dropout(0.5)
    ])

    input = renom.Variable(np.random.random((10, 10, 10, 10)))
    m = _run_onnx(tmpdir, model, input)

    # check input
    id_input, = m.graph.node[0].input
    assert get_shape(m.graph.input[0]) == input.shape

    # check output
    assert get_shape(m.graph.output[0]) == input.shape

    attrs = dict((a.name, a) for a in m.graph.node[0].attribute)
    assert attrs['ratio'].f == 0.5


def test_reshape(tmpdir):
    model = rm.Sequential([
        rm.Flatten()
    ])

    input = renom.Variable(np.random.random((10, 10, 10, 10)))
    m = _run_onnx(tmpdir, model, input)

    # check input
    id_input, id_shape = m.graph.node[0].input
    assert get_shape(m.graph.input[0]) == input.shape

    inis = load_initializer(m.graph.initializer)
    _test_initializer(inis, id_shape, [10, -1])

    # check output
    assert get_shape(m.graph.output[0]) == (10, 1000)
