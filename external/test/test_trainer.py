import os
import pytest

import numpy as np
import renom as rm
from renom.utility.distributor import NdarrayDistributor
from renom.utility.trainer import *


def test_trainer():
    class NN(rm.Model):
        def __init__(self):
            super(NN, self).__init__()
            self.params.value1 = rm.Variable(np.array([1., 2., 3., 4.]))
            self.params.value2 = rm.Variable(np.array([1., 2., 3., 4.]))

        def forward(self, v):
            return v * self.params.value1 * self.params.value2

    distributor = NdarrayDistributor(
        np.array([[1., 2., 3., 4.], [1., 2., 3., 4.]]),
        np.array([[1., 2., 3., 4.], [1., 2., 3., 4.]]))

    trainer = Trainer(NN(), num_epoch=10, loss_func=rm.softmax_cross_entropy,
                      batch_size=100, optimizer=rm.Sgd())

    l = set()

    @trainer.events.start
    def start(trainer):
        l.add('start')

    @trainer.events.start_epoch
    def start_epoch(trainer):
        l.add('start_epoch')

    @trainer.events.forward
    def forward(trainer):
        l.add('forward')

    @trainer.events.backward
    def backward(trainer):
        l.add('backward')

    @trainer.events.updated
    def updated(trainer):
        l.add('updated')

    @trainer.events.end_epoch
    def end_epoch(trainer):
        l.add('end_epoch')

    trainer.train(distributor)
    assert l == set(['start', 'start_epoch', 'forward', 'backward', 'updated', 'end_epoch'])
