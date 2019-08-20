#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import pytest
from renom.cuda import use_cuda
from renom.core import Variable
import renom as rm
import test_utility


@test_utility.skipgpu
@pytest.mark.skipif(np.lib.NumpyVersion(np.__version__) < '1.13.0', reason='na')
def test_gpu_node_neg():
    with use_cuda():
        g1 = Variable(np.array([1., 2.]))
        g2 = -g1
        assert np.allclose(g2, [-1, -2])
        assert not np.allclose(g2, [-1, -3])

        g3 = -g1 * 2
        assert np.allclose(g3, [-2, -4])
        assert not np.allclose(g3, [-3, -4])


def test_grad():
    class D(Variable):
        def _update_diff(self, context, dy, **kwargs):
            ready = context.add(self, dy)
            print(id(self), ready)

            if ready:
                diff = context.get(self)
                self.backward(context, diff, **kwargs)

    g1 = D(np.array([1., 2.]))
    g2 = D(np.array([1., 2.]))
    g3 = D(np.array([1., 2.]))
    g4 = g1 + g2
    g5 = g3 + g4
    g6 = g1 + g2 + g3 + g4 + g5
    g7 = g6 + g5

    print([id(g) for g in (g1, g2, g3, g4, g5, g6, g7)])

    g = g6.grad(np.array([1., 2.]))
    print(g._refcounts)
    print(g._backwards)
    assert len(g._refcounts) == 9

    assert g._refcounts[id(g1)] == g._backwards[id(g1)]
    assert g._refcounts[id(g2)] == g._backwards[id(g2)]
    assert g._refcounts[id(g3)] == g._backwards[id(g3)]
    assert g._refcounts[id(g4)] == g._backwards[id(g4)]
    assert g._refcounts[id(g5)] == g._backwards[id(g5)]


def test_grad2():
    class D(Variable):
        pass

    a = D(np.array([1., 2.]))
    b = D(np.array([1., 2.]))
    c = a + b
    d = a + c
    e = b + c
    f = d + e

    print([id(g) for g in (a, b, c, d, e, f)])

    g = f.grad(np.array([1., 2.]))
    print(g._refcounts)
    print(g._backwards)
