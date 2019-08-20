import time
import numpy as np
import renom as rm
from renom.cuda import cuda


def prof_add():
    cuda.set_cuda_active(True)
    a = np.random.rand(1000, 1000).astype(np.float32)
    b = np.random.rand(1000, 1000).astype(np.float32)
    c = np.random.rand(1, 1000).astype(np.float32)
    ga = rm.Variable(a)
    gb = rm.Variable(b)
    gc = rm.Variable(c)
    start_t = time.time()
    for _ in range(1000):
        ga + gb * gc
    print("took time %f" % (time.time() - start_t))

    start_t = time.time()
    for _ in range(1000):
        a + b * c
    print("took time %f" % (time.time() - start_t))


if __name__ == "__main__":
    prof_add()
