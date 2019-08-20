import time
import numpy as np
import renom as rm
from renom.cuda import cuda
from renom.optimizer import Sgd


def exp_convolution1():
    np.random.seed(10)
    # Caused by CUDNN_CONVOLUTION_FWD_ALGO_GEMM is not deterministic.
    # 1724.07080078 GPU
    # 1715.86767578 CPU
    cuda.set_cuda_active(True)
    a = np.random.randn(8 * 2, 64, 32, 32).astype(np.float32)
    b = np.random.randn(8 * 2, 32, 28, 28).astype(np.float32)
    layer1 = rm.Conv2d(channel=32, input_size=a.shape[1:])
    layer2 = rm.Conv2d(channel=32, input_size=(32, 30, 30))

    ga = rm.Variable(a, auto_update=False)
    gb = rm.Variable(b, auto_update=False)

    opt = Sgd(0.0001, momentum=0.0)
    start_t = time.time()
    for _ in range(100):
        loss = rm.Sum((layer2(rm.Relu(layer1(ga))) - gb)**2) / 8
        loss.ensure_cpu()
        grad = loss.grad()
        grad.update(opt)
        print(loss)
    print(time.time() - start_t)


def exp_convolution2():
    np.random.seed(10)
    cuda.set_cuda_active(True)
    a = np.random.randn(8, 3, 12, 12).astype(np.float32)
    b = np.random.randn(8, 16, 10, 10).astype(np.float32)
    layer1 = rm.Conv2d(channel=16, input_size=a.shape[1:])

    ga = rm.Variable(a, auto_update=False)
    gb = rm.Variable(b, auto_update=False)

    opt = Sgd(0.001, momentum=0.3)
    start_t = time.time()
    for _ in range(100000):
        loss = rm.Sum((rm.Sigmoid(layer1(ga)) - gb)**2) / 8
        loss.ensure_cpu()
        print(loss)
        grad = loss.grad()
        grad.update(opt)
        del loss
    print(time.time() - start_t)


def exp_dense():
    np.random.seed(10)
    cuda.set_cuda_active(False)
    a = np.random.rand(32, 320).astype(np.float32)
    b = np.random.rand(32, 80).astype(np.float32)
    layer1 = rm.Dense(input_size=320, output_size=100)
    layer2 = rm.Dense(input_size=100, output_size=80)
    ga = rm.Variable(a, auto_update=False)
    gb = rm.Variable(b, auto_update=False)
    opt = Sgd(0.01, momentum=0.3)
    start_t = time.time()

    for _ in range(500):
        loss = rm.Sum((layer2(rm.Sigmoid(layer1(ga))) - gb)**2) / 32
        loss.ensure_cpu()
        print(loss)
        grad = loss.grad()
        grad.update(opt)
    print(time.time() - start_t)


if __name__ == "__main__":
    #     exp_convolution1()
    exp_dense()
