#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from itertools import product
import renom.cuda as cuda
from renom.utility.reinforcement.replaybuffer import ReplayBuffer
from renom.utility.searcher import GridSearcher, RandomSearcher, BayesSearcher

skipgpu = pytest.mark.skipif(not cuda.has_cuda(), reason="cuda is not installed")
skipmultigpu = pytest.mark.skipif(
    not cuda.has_cuda() or (cuda.cuGetDeviceCount() < 2),
    reason="Number of gpu card is less than 2.")

np.random.seed(101)
eps = np.sqrt(np.finfo(np.float32).eps)


def auto_diff(function, node, *args):
    loss = function(*args)
    return loss.grad().get(node)


def numeric_diff(function, node, *args):
    shape = node.shape
    diff = np.zeros_like(node)

    if True:  # 5 point numerical diff
        coefficients1 = [1, -8, 8, -1]
        coefficients2 = [-2, -1, 1, 2]
        c = 12
    else:    # 3 point numerical diff
        coefficients1 = [1, -1]
        coefficients2 = [1, -1]
        c = 2

    node.to_cpu()
    node.setflags(write=True)
    for nindex in np.ndindex(shape):
        loss = 0
        for i in range(len(coefficients1)):
            dx = eps * node[nindex] if node[nindex] != 0 else eps
            node[nindex] += coefficients2[i] * dx
            node.to_cpu()

            ret_list = function(*args) * coefficients1[i]
            ret_list.to_cpu()
            node[nindex] -= coefficients2[i] * dx

            loss += ret_list
            loss.to_cpu()

        v = loss / (dx * c)
        v.to_cpu()
        diff[nindex] = v
    diff.to_cpu()
    return diff


@pytest.mark.parametrize("action_size, state_size, data_size", [
    [(2, ), (3, ), 4],
    [(2, ), (3, 2), 4],
])
def test_replaybuffer(action_size, state_size, data_size):
    buffer = ReplayBuffer(action_size, state_size, data_size)
    a, p, s, r, t = [[] for _ in range(5)]
    for i in range(data_size):
        a.append(np.random.rand(*action_size))
        p.append(np.random.rand(*state_size))
        s.append(np.random.rand(*state_size))
        r.append(np.random.rand(1))
        t.append(np.random.rand(1).astype(np.bool))
        buffer.store(p[-1], a[-1], r[-1], s[-1], t[-1])

    data = buffer.get_minibatch(data_size, False)
    for i in range(data_size):
        assert np.allclose(p[i], data[0][i])
        assert np.allclose(a[i], data[1][i])
        assert np.allclose(r[i], data[2][i])
        assert np.allclose(s[i], data[3][i])
        assert np.allclose(t[i], data[4][i])


@pytest.mark.parametrize("param_space", [
    {"a": [1, 2], "b":[3, 4]},
    {"a": [1, 2], "b":[3, 4], "c":[4, 5, 3]},
])
def test_grid_searcher(param_space):
    searcher = GridSearcher(param_space)
    for params in searcher.suggest():
        searcher.set_result(np.sum(list(params.values())))
    assert searcher.best()[0][1] == \
        np.min(list(map(lambda x: np.sum(x), product(*list(param_space.values())))))


@pytest.mark.parametrize("param_space", [
    {"a": [1, 2], "b":[3, 4]},
    {"a": [1, 2], "b":[3, 4], "c":[4, 5, 3]},
])
def test_random_searcher(param_space):
    searcher = RandomSearcher(param_space)
    for params in searcher.suggest(100):
        searcher.set_result(np.sum(list(params.values())))
    assert searcher.best()[0][1] == \
        np.min(list(map(lambda x: np.sum(x), product(*list(param_space.values())))))


@pytest.mark.skip
@pytest.mark.parametrize("param_space", [
    {"a": [1, 2, 3], "b":[-1, 3, 4, 5]},
    {"a": [1, 2], "b":[3, 4, -1], "c":[4, 5, 3]},
])
def test_bayes_searcher(param_space):
    searcher = BayesSearcher(param_space)
    for params in searcher.suggest(random_iter=5):
        searcher.set_result(np.sum(list(params.values())))
    assert searcher.best()[0][1] == \
        np.min(list(map(lambda x: np.sum(x), product(*list(param_space.values())))))
