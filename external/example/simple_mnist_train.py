#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import matplotlib.pyplot as plt
import time
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

from renom import *
from renom.core import DEBUG_NODE_STAT, DEBUG_GRAPH_INIT, DEBUG_NODE_GRAPH
from renom.utility.trainer import Trainer
from renom.utility.distributor import NdarrayDistributor

DEBUG_GRAPH_INIT(True)
cuda.set_cuda_active(True)

mnist = fetch_mldata('MNIST original', data_home="dataset")

X = mnist.data.astype(np.float32)
X /= X.max()
y = mnist.target

labels = LabelBinarizer().fit_transform(y).astype(np.float32)

X = np.concatenate([X] * 5)
labels = np.concatenate([labels] * 5)


class MNist(Model):
    def __init__(self):
        super(MNist, self).__init__()
        self.layer1 = Dense(output_size=2000)
        self.layer2 = Dense(output_size=2000)
        self.layer3 = Dense(output_size=2000)
        self.layer4 = Dense(output_size=10)
        self.bn = BatchNormalize()

    def forward(self, x):
        ret = self.layer2(relu(self.layer1(x)))
        return self.layer4(relu(self.layer3(ret)))


train_dist, test_dist = NdarrayDistributor(X, labels).split(0.9)

num_gpu = 1  # cuda.cuGetDeviceCount() or 1
trainer = Trainer(MNist(), num_epoch=10, loss_func=softmax_cross_entropy,
                  batch_size=20000, optimizer=Sgd(), num_gpu=num_gpu)

loss = 0
learning_curve = []
test_learning_curve = []


@trainer.events.start_epoch
def start_epoch(trainer):
    global loss, start_t
    loss = 0
    start_t = time.time()


@trainer.events.updated
def updated(trainer):
    global loss
    loss += trainer.losses[0]


@trainer.events.end_epoch
def end_epoch(trainer):
    train_loss = loss / (trainer.nth + 1)
    learning_curve.append(train_loss)

    test_loss = 0

    trainer.model.set_models(inference=True)
    for i, (x, y) in enumerate(test_dist.batch(128)):
        test_result = trainer.test(x)
        test_loss += trainer.loss_func(test_result, y)
    trainer.model.set_models(inference=False)

    test_loss /= i + 1
    test_learning_curve.append(test_loss)

    print("epoch %03d train_loss:%f test_loss:%f took time:%f" %
          (trainer.epoch, train_loss, test_loss, time.time() - start_t))


trainer.train(train_dist)

# Test
trainer.model.set_models(inference=True)
ret = np.vstack((trainer.test(x) for x, _ in test_dist.batch(128, shuffle=False)))
predictions = np.argmax(np.array(ret), axis=1)
label = np.argmax(test_dist.y, axis=1)

print(confusion_matrix(label, predictions))
print(classification_report(label, predictions))

plt.hold(True)
plt.plot(learning_curve, linewidth=3, label="train")
plt.plot(test_learning_curve, linewidth=3, label="test")
plt.ylabel("error")
plt.xlabel("epoch")
plt.legend()
plt.show()

DEBUG_NODE_STAT()
