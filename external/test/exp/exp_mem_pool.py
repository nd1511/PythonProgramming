
from renom.cuda.cuda import *
from renom.cuda.cublas import *
from renom.cuda.cuda_base import *


class Mempool(object):

    size = None
    nbytes = None
    ptr = None
    available = None

    def __init__(self, nbytes, ptr):
        self.ptr = ptr
        self.nbytes = nbytes
        self.available = True


class gpu_allocator(object):

    pools = []

    @classmethod
    def malloc(cls, nbytes):
        pool = cls.getAvailablePool(nbytes)
        if pool is None:
            ptr = cuMalloc(nbytes)
            pool = Mempool(nbytes=nbytes, ptr=ptr)
            cls.pools.append(pool)
        return ptr

    @classmethod
    def memset(cls, ptr, value, nbytes):
        ptr = cuMemset(ptr, value, nbytes)

    @classmethod
    def free(cls, ptr):
        cuFree(ptr)

    @classmethod
    def memcpyH2D(cls, cpu_ptr, gpu_ptr, nbytes):
        cuMemcpyH2D(cpu_ptr.flatten(), gpu_ptr, nbytes)

    @classmethod
    def memcpyD2D(cls, gpu_ptr1, gpu_ptr2, nbytes):
        cuMemcpyD2D(gpu_ptr1, gpu_ptr2, nbytes)

    @classmethod
    def memcpyD2H(cls, gpu_ptr, cpu_ptr, nbytes):
        shape = cpu_ptr.shape
        cpu_ptr = cpu_ptr.reshape(-1)
        cuMemcpyD2H(gpu_ptr, cpu_ptr, nbytes)
        cpu_ptr.reshape(shape)

    @classmethod
    def getAvailablePool(cls, size):
        pool = None
        for p in cls.pools:
            if size == p.nbytes and p.available:
                p = pool
        return pool
