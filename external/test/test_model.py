import os
import pytest

import numpy as np
import renom as rm
from renom.core import Variable, to_value
from renom import DEBUG_GRAPH_INIT, DEBUG_NODE_GRAPH
from renom.cuda import set_cuda_active
from renom.cuda import use_device
import test_utility

set_cuda_active(True)


class NN(rm.Model):
    def __init__(self):
        super(NN, self).__init__()
        self.params.value1 = Variable(np.array([1., 2., 3., 4.]))
        self.params.value2 = Variable(np.array([1., 2., 3., 4.]))

    def forward(self, v):
        return v * self.params.value1 * self.params.value2


def test_train():

    nn = NN()

    with nn.train():
        ret = nn(np.array([1., 2., 3., 4.]))

    grad = ret.grad(1)
    grad.update()

    assert np.allclose(nn.params.value1.as_ndarray(), [0., -2., -6., -12.])
    assert np.allclose(nn.params.value2.as_ndarray(), [0., -2., -6., -12.])


def test_train2():

    nn = NN()
    nn2 = NN()

    with nn2.train():
        ret = nn(np.array([1., 2., 3., 4.]))
        ret2 = nn2(ret)

    grad = ret2.grad(1)
    grad.update()

    assert np.allclose(nn.params.value1.as_ndarray(), [1., 2., 3., 4.])
    assert np.allclose(nn.params.value2.as_ndarray(), [1., 2., 3., 4.])

    assert np.allclose(nn2.params.value1.as_ndarray(), [0., -14., -78., -252.])
    assert np.allclose(nn2.params.value2.as_ndarray(), [0., -14., -78., -252.])


def test_not_train():
    nn = NN()

    ret = nn(np.array([1, 2, 3, 4]))
    assert not list(ret.attrs.get_attrs())

    grad = ret.grad(1)
    grad.update()

    nn.params.value1.to_cpu()
    nn.params.value2.to_cpu()

    assert np.allclose(np.array([1, 2, 3, 4]), nn.params.value1.as_ndarray())
    assert np.allclose(np.array([1, 2, 3, 4]), nn.params.value2.as_ndarray())


def test_prevent():
    nn = NN()

    with nn.train():
        ret = nn(np.array([1, 2, 3, 4]))

    assert list(ret.attrs.get_attrs())
    grad = ret.grad(1)
    with nn.prevent_update():
        grad.update()

    assert np.allclose(nn.params.value1.as_ndarray(), [1, 2, 3, 4])
    assert np.allclose(nn.params.value2.as_ndarray(), [1, 2, 3, 4])


try:
    import h5py
    has_h5py = True
except ImportError:
    has_h5py = False

skiph5py = pytest.mark.skipif(not has_h5py, reason="h5py is not installed")


@skiph5py
def test_save(tmpdir_factory):

    class NN2(rm.Model):
        def __init__(self):
            super(NN2, self).__init__()
            self.layer1 = rm.Dense(output_size=2)
            self.layer2 = rm.Dense(output_size=2)
            self.bn = rm.BatchNormalize()

        def forward(self, x):
            return self.layer2(self.bn(rm.relu(self.layer1(x))))

    class NN3(rm.Model):
        SERIALIZED = ('AAA', 'BBB')

        def __init__(self):
            super(NN3, self).__init__()
            self.layer1 = NN2()
            self.layer2 = NN2()
            self.AAA = 0

        def forward(self, x):
            return self.layer2(rm.relu(self.layer1(x)))

    nn = NN3()
    with nn.train():
        result = nn(np.random.rand(2, 2))
        l = rm.softmax_cross_entropy(result, np.random.rand(2, 2))

    grad = l.grad()
    opt = rm.Sgd()
    grad.update(opt)

    nn.layer1.layer1.params.b._auto_update = False

    d = tmpdir_factory.mktemp('h5')
    fname = os.path.join(str(d), 'aaa')
    nn.AAA = 9999
    nn.save(fname)

    nn2 = NN3()
    nn2.load(fname)

    assert np.allclose(nn.layer1.layer1.params.w, nn2.layer1.layer1.params.w)
    assert np.allclose(nn.layer1.layer1.params.b, nn2.layer1.layer1.params.b)
    assert np.allclose(nn.layer1.layer2.params.w, nn2.layer1.layer2.params.w)
    assert np.allclose(nn.layer1.layer2.params.b, nn2.layer1.layer2.params.b)

    assert np.allclose(nn.layer2.layer1.params.w, nn2.layer2.layer1.params.w)
    assert np.allclose(nn.layer2.layer1.params.b, nn2.layer2.layer1.params.b)
    assert np.allclose(nn.layer2.layer2.params.w, nn2.layer2.layer2.params.w)
    assert np.allclose(nn.layer2.layer2.params.b, nn2.layer2.layer2.params.b)

    assert nn2.layer1.layer1.params.w._auto_update
    assert not nn2.layer1.layer1.params.b._auto_update
    assert nn2.AAA == 9999


def test_weight_decay():
    set_cuda_active(False)

    def add_weight_decay(weighted_model, decay):
        reg = rm.sum(weighted_model.params.w * weighted_model.params.w)
        return reg * (decay / 2)

    test_decay = 0.25
    input = np.random.rand(2, 2)
    unweighted_model = rm.Dense(2, input_size=(2,))
    weighted_model = rm.Dense(2, input_size=(2,), weight_decay=test_decay)
    unweighted_model.params['w'].setflags(write=True)
    unweighted_model.params['w'][...] = weighted_model.params['w'][...]

    set_cuda_active(True)

    with unweighted_model.train(), weighted_model.train():
        unweighted_loss = rm.sum(unweighted_model(input))
        weighted_loss = rm.sum(weighted_model(input))
        added_loss = rm.sum(unweighted_loss + add_weight_decay(unweighted_model, test_decay))
    n_1 = unweighted_loss.grad(weight_decay=test_decay, detach_graph=False).get(
        unweighted_model.params["w"])
    n_2 = weighted_loss.grad(detach_graph=False).get(weighted_model.params["w"])
    n_3 = added_loss.grad(detach_graph=False).get(unweighted_model.params["w"])
    map(lambda x: x.to_cpu(), [n_1, n_2, n_3])
    try:
        assert np.allclose(n_1, n_2)
        assert np.allclose(n_1, n_3)
    except AssertionError:
        print("Error in weight decay")
        print(n_1)
        print(n_2)
        print(n_3)
        assert False


def test_update():
    nn = rm.Dense(2)
    nn2 = rm.Dense(2)
    with nn.train():
        ret = nn(np.random.rand(2, 2))
        loss = rm.softmax_cross_entropy(ret, np.random.rand(2, 2))

    cur = nn.params.w.copy()
    grad = loss.grad(np.array([1]))

    grad.update(models=[nn2])
    assert np.allclose(cur.as_ndarray(), nn.params.w)

    grad.update(models=[nn])
    assert np.allclose(cur.as_ndarray() - grad.get(nn.params.w), nn.params.w.as_ndarray())


@test_utility.skipgpu
def test_multi_gpu():
    from renom.cuda import cuGetDeviceCount

    class NN2(rm.Model):
        def __init__(self):
            super(NN2, self).__init__()
            self.layer1 = rm.Dense(output_size=2)
            self.layer2 = rm.Dense(output_size=2)

        def forward(self, x):
            return self.layer2(rm.relu(self.layer1(x)))

        def weight_initiallize(self, input_size):
            self.layer1.weight_initiallize(input_size)
            self.layer2.weight_initiallize(input_size)

    nn = NN2()
    nn.set_gpu(0)
    nn.weight_initiallize((2,))

    nn2 = NN2()
    nn2.set_gpu(cuGetDeviceCount() - 1)

    for i in range(2):
        nn2.copy_params(nn)
        x = np.random.rand(100, 2)
        with nn.train():
            ret1 = nn(x[:50])

        with use_device(nn.device_id):
            loss1 = rm.softmax_cross_entropy(ret1, np.random.rand(50, 2))

        with nn2.train():
            ret2 = nn2(x[50:])

        with use_device(nn2.device_id):
            loss2 = rm.softmax_cross_entropy(ret2, np.random.rand(50, 2))

        nn.sync()
        nn2.sync()

        grad1 = loss1.grad()

        with use_device(nn2.device_id):
            grad2 = loss2.grad()

        grad2.get(nn2.layer1.params.w)
        org_l1_w = grad1.get(nn.layer1.params.w)

        nn.join_grads(grad1, [(nn2, grad2)])

        assert np.allclose(grad1.get(nn.layer1.params.w),
                           org_l1_w + grad2.get(nn2.layer1.params.w).copy())

        grad1.update(models=[nn])
