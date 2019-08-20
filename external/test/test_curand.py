import numpy as np
import renom as rm
from renom.cuda import *
import pytest
import test_utility


@test_utility.skipgpu
@pytest.mark.parametrize("generator_type", [
    "rand_normal",
    "rand_bernoulli",
    "rand_uniform",
])
def test_curand_equal(generator_type):
    set_cuda_active(True)
    X = rm.GPUValue(shape=(4, 4))
    Y = rm.GPUValue(shape=(4, 4))
    gen = curand_generator()
    random_func = getattr(gen, generator_type)
    gen.set_seed(30)
    random_func(X)
    X = rm.Node(X)
    gen.set_seed(30)
    random_func(Y)
    Y = rm.Node(Y)
    X.to_cpu()
    Y.to_cpu()
    assert np.allclose(X, Y), "Not Equal\nX=\n{}\nY=\n{}\n".format(X, Y)


@test_utility.skipgpu
@pytest.mark.parametrize("generator_type", [
    "rand_normal",
    "rand_bernoulli",
    "rand_uniform",
])
def test_curand_nequal(generator_type):
    set_cuda_active(True)
    X = rm.GPUValue(shape=(4, 4))
    Y = rm.GPUValue(shape=(4, 4))
    gen = curand_generator()
    random_func = getattr(gen, generator_type)
    gen.set_seed(30)
    random_func(X)
    X = rm.Node(X)
    gen.set_seed(45)
    random_func(Y)
    Y = rm.Node(Y)
    X.to_cpu()
    Y.to_cpu()
    assert not np.allclose(X, Y), "Equal\nX=\n{}\nY=\n{}\n".format(X, Y)
