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
from renom.utility.initializer import Gaussian
from renom.utility.trainer import Trainer
from renom.utility.distributor import NdarrayDistributor

cuda.set_cuda_active(False)

mnist = fetch_mldata('MNIST original', data_home="dataset")

X = mnist.data.astype(np.float32)
X /= X.max()

X = X.reshape(-1, 28, 28)
y = mnist.target

labels = LabelBinarizer().fit_transform(y).astype(np.float32)


class MNist(Model):
    def __init__(self):
        super(MNist, self).__init__()
        self.layer0 = Dense(output_size=50)
        self.layer1 = PeepholeLstm(output_size=50)
        self.layer2 = Dense(output_size=10)

    def forward(self, x):
        self.truncate()
        ret = 0
        for i in range(28):
            lstm = self.layer1(x[:, i])
            ret = self.layer2(lstm)
        return ret


train_dist, test_dist = NdarrayDistributor(X, labels).split(0.9)

trainer = Trainer(MNist(), num_epoch=10, loss_func=softmax_cross_entropy,
                  batch_size=128, optimizer=Adam())

trainer.train(train_dist)

# Test
trainer.model.set_models(inference=True)
ret = trainer.test(test_dist.data()[0])
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
