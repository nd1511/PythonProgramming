import pytest


@pytest.fixture(params=[False, True])
def use_gpu(request):
    """
    Gpu switch for test.
    """
    yield request.param


@pytest.fixture(params=[False, True])
def ignore_bias(request):
    """
    Bias switch for test.
    """
    yield request.param
