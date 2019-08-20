
import time
import numpy as np
import pytest

import renom.cuda.cuda_base as cm


@pytest.mark.skip('FIXME')
def test_stream():
    cpu_array1 = np.random.rand(1024 * 32).astype(np.float32)
    cpu_array2 = np.random.rand(1024 * 32).astype(np.float32)
    cpu_array3 = np.empty_like(cpu_array1)
    cpu_array4 = np.empty_like(cpu_array1)

    gpu_array1 = cm.cuMalloc(cpu_array1.nbytes)
    gpu_array2 = cm.cuMalloc(cpu_array2.nbytes)

    cm.cuMemcpyH2D(cpu_array1.flatten().copy(), gpu_array1, cpu_array1.nbytes)
    cm.cuMemcpyH2D(cpu_array2.flatten().copy(), gpu_array2, cpu_array1.nbytes)

    stream1 = cm.cuCreateStream()
    stream2 = cm.cuCreateStream()

    start_t = time.time()
    for _ in range(100):
        cm.cuMemcpyH2DAsync(cpu_array1, gpu_array1, cpu_array1.nbytes, stream1)
        cm.cuMemcpyD2HAsync(gpu_array1, cpu_array3, cpu_array1.nbytes, stream2)

        cm.cuMemcpyH2DAsync(cpu_array2, gpu_array2, cpu_array2.nbytes, stream1)
        cm.cuMemcpyD2HAsync(gpu_array2, cpu_array4, cpu_array2.nbytes, stream2)
    print(time.time() - start_t)

    start_t = time.time()
    for _ in range(100):
        cm.cuMemcpyH2D(cpu_array1, gpu_array1, cpu_array1.nbytes)
        cm.cuMemcpyD2H(gpu_array1, cpu_array3, cpu_array1.nbytes)
        cm.cuMemcpyH2D(cpu_array2, gpu_array2, cpu_array2.nbytes)
        cm.cuMemcpyD2H(gpu_array2, cpu_array4, cpu_array2.nbytes)
    print(time.time() - start_t)


@pytest.mark.skip('FIXME')
def test_context():
    cm.cuCreateCtx(0)
    cm.cuGetDeviceCxt()
    cm.cuGetDeviceProperty(0)


@pytest.mark.skip('FIXME')
def test_allocator():
    cm.mem_pool()
#    time.sleep(100)
