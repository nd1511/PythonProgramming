#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import renom
import renom.cuda
from renom.cuda import thrust_float
from renom.cuda import thrust_double
from renom.core import GPUValue, Node


def test_negate_float():
    arr = np.array(np.random.rand(4,))
    v = GPUValue(arr)
    dest = v.empty_like_me()

    thrust_float.negate(v, dest)

    assert np.allclose(arr * -1, dest.to_array())


# def test_negate_double():
#     arr = np.array(np.random.rand(4,), dtype=np.float64)
#     v = GPUValue(arr)
#     dest = v.empty_like_me()
#     thrust_double.negate(v, dest)
#     assert np.allclose(arr*-1, dest.to_array())
