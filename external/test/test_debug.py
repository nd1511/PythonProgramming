#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.core import Variable
from renom import DEBUG_NODE_STAT, DEBUG_GRAPH_INIT, DEBUG_NODE_GRAPH, SET_MODEL_GRAPH, BUILD_MODEL_GRAPH
from renom import operation as O
import renom as R


def test_node_dump():
    DEBUG_GRAPH_INIT(True)

    a = Variable(np.array([1, 2, 3, 4, 5]))
    b = Variable(np.array([1, 2, 3, 4, 5]))
    c = a + b  # NOQA

    d = a + b * 2  # NOQA

    DEBUG_NODE_STAT()
    # DEBUG_NODE_GRAPH()

    DEBUG_GRAPH_INIT(False)


def test_node_clear():
    DEBUG_GRAPH_INIT(True)

    a = Variable(np.random.rand(2, 2).astype(np.float32))
    b = Variable(np.random.rand(2, 2).astype(np.float32))

    layer = R.Lstm(2)

    c = layer(O.dot(a, b))  # NOQA

    DEBUG_NODE_STAT()
#    DEBUG_NODE_GRAPH()
    DEBUG_GRAPH_INIT(False)


def test_graph():
    try:
        import graphviz
    except ImportError:
        return

    SET_MODEL_GRAPH(True)

    class MNist(R.Model):
        def __init__(self):
            super(MNist, self).__init__()
            self.layer1 = R.PeepholeLstm(output_size=50)
            self.layer2 = R.Dense(output_size=50)

        def forward(self, x):
            self.truncate()
            ret = 0
            for i in range(2):
                lstm = self.layer1(x[:, 0])
                ret += self.layer2(lstm)
            return ret

    model = MNist()
    v = model(np.random.rand(10, 28, 28))
    g = BUILD_MODEL_GRAPH(model, v)
    assert g

    SET_MODEL_GRAPH(False)
