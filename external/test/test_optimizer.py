import numpy as np
from renom.core import Variable, Node, to_value
from renom.operation import dot
from renom.config import precision
from renom.optimizer import *
from renom.utility.initializer import GlorotNormal
from renom.cuda import set_cuda_active
from renom.layers.loss.mean_squared_error import mean_squared_error
from renom.layers.function.parameterized import Parametrized
import test_utility


class Weighted_test_model(Parametrized):
    def __init__(self, output_size, input_size=None, initializer=GlorotNormal()):
        self._output_size = output_size
        self._initializer = initializer
        super(Weighted_test_model, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        size_i = input_size[0] if isinstance(input_size, tuple) else input_size
        size_o = self._output_size
        self.params = {
            "w": Variable(self._initializer((size_i, size_o)), auto_update=True)}

    def forward(self, x):
        return dot(x, self.params["w"])


def close(GPU, CPU):
    print('GPU =')
    print(to_value(GPU))
    print('CPU =')
    print(to_value(CPU))
    assert np.allclose(to_value(GPU), to_value(CPU), atol=1e-4, rtol=1e-3)


def create_data():
    x = np.arange(10).reshape(10, 1)
    mult = 2
    noise_mult = 5
    y = x * mult + np.random.rand(*x.shape) * noise_mult
    return x, y


def optimizer_check(optimizer=None):
    np.random.seed(10)
    x, y = create_data()

    model = Weighted_test_model(1)
    epochs = 200

    print('\033[K\033[1;32m')
    print()

    for e in range(epochs):
        for j in range(len(x)):
            x_batch = x[j:j + 1, :]
            y_batch = y[j:j + 1, :]
            with model.train():
                z = model(x_batch)
                l = mean_squared_error(z, y_batch) / (4 * (x_batch**2) + 1)
            l.grad().update(optimizer)

        if (e + 1) % 100 is 0:
            print('\033[2A')
            print('\033[KFinished \033[1;31mEpoch #{:d} \033[1;32mfor optimizer \033[1;31m{}\033[1;32m'.format(
                e + 1, type(optimizer)))

    print('\033[0m\033[1A')

    assert 2 - np.squeeze(model.params['w'].as_ndarray()) < 1.0, "Optimal weight value is {:d}, learned value of {:f}".format(
        2, np.squeeze(model.params['w'].as_ndarray()))


def gpu_check(opt):
    node = Variable(np.array(np.random.rand(3, 3, 3, 3), dtype=precision))
    grad = Variable(np.array(np.random.rand(3, 3, 3, 3), dtype=precision))

    set_cuda_active(False)
    for _ in range(3):
        opt(grad, node)
    dy_cpu = opt(grad, node)
    assert isinstance(dy_cpu, Node)
    opt.reset()

    set_cuda_active(True)
    for _ in range(3):
        opt(grad, node)
    dy_gpu = opt(grad, node)
    assert isinstance(dy_gpu, GPUValue)
    dy_gpu = Node(dy_gpu)
    dy_gpu.to_cpu()

    close(dy_gpu, dy_cpu)


def test_sgd_correct():
    optimizer_check(Sgd())


@test_utility.skipgpu
def test_sgd_gpu():
    gpu_check(Sgd())


def test_Adagrad_correct():
    optimizer_check(Adagrad())


@test_utility.skipgpu
def test_Adagrad_gpu():
    gpu_check(Adagrad())


def test_Rmsprop_correct():
    optimizer_check(Rmsprop())


@test_utility.skipgpu
def test_Rmsprop_gpu():
    gpu_check(Rmsprop())


def test_Adam_correct():
    optimizer_check(Adam())


@test_utility.skipgpu
def test_Adam_gpu():
    gpu_check(Adam())
