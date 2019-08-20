#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import matplotlib.pyplot as plt
import copy
import time
import numpy as np
from multiprocessing import Pipe, Process

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

import renom as rm
from renom.cuda import cuda
from renom.optimizer import Sgd, Adam
from renom.core import DEBUG_NODE_STAT, DEBUG_GRAPH_INIT, DEBUG_NODE_GRAPH
from renom.operation import sum


DEBUG_GRAPH_INIT(True)

np.random.seed(10)

cuda.set_cuda_active(True)

mnist = fetch_mldata('MNIST original', data_home="dataset")

X = mnist.data
y = mnist.target

X = X.astype(np.float32)
X /= X.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
labels_train = LabelBinarizer().fit_transform(y_train).astype(np.float32)
labels_test = LabelBinarizer().fit_transform(y_test).astype(np.float32)


class MNist(rm.Model):
    def __init__(self):
        super(MNist, self).__init__()
        self.layer1 = rm.Dense(output_size=100)
        self.layer2 = rm.Dense(output_size=10)

    def forward(self, x):
        return self.layer2(rm.relu(self.layer1(x)))


epoch = 10
batch = 100
count = 0

learning_curve = []
test_learning_curve = []

opt = Adam()
opt = Sgd()

N = len(X_train)

nn = MNist()

for i in range(epoch):
    start_t = time.time()
    perm = np.random.permutation(N)
    loss = 0
    for j in range(0, N // batch):
        train_batch = X_train[perm[j * batch:(j + 1) * batch]]
        responce_batch = labels_train[perm[j * batch:(j + 1) * batch]]

        with nn.train():
            result = nn(train_batch)
            ls = rm.softmax_cross_entropy(result, responce_batch)

        ls.to_cpu()

        grad = ls.grad()
        grad.update(opt)
        loss += ls

    train_loss = loss / (N // batch)
    train_loss.to_cpu()
    test_loss = rm.softmax_cross_entropy(nn(X_test), labels_test)
    test_loss.to_cpu()

    test_learning_curve.append(test_loss)
    learning_curve.append(train_loss)
    print("epoch %03d train_loss:%f test_loss:%f took time:%f" %
          (i, train_loss, test_loss, time.time() - start_t))

ret = nn(X_test)
ret.to_cpu()
predictions = np.argmax(np.array(ret), axis=1)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

plt.hold(True)
plt.plot(learning_curve, linewidth=3, label="train")
plt.plot(test_learning_curve, linewidth=3, label="test_old")
plt.ylabel("error")
plt.xlabel("epoch")
plt.legend()
plt.show()

loss = None
nn = None
grad = None
test_learning_curve = learning_curve = None
DEBUG_NODE_STAT()
