# GAN, Generative Adversarial Network
# we use GANs with images

# In a GAN, a G competes with a D
# G = Generative model and D = Discriminative model

# two models compete, the generator and the discriminator

# the generator creates fake data, generative model
# the discriminator classifies the data as fake or true
# the discriminator has to find the fake data produced by the generator

# minmax game, two models compete in a minimax game
# the one model tries to minimize a function and the other to maximize the function

# min max (E_{x ~ Pdata} {log(D(x))} + E_{Z ~ P(Z)} {log(1-D(G(Z)))})

# min => generator, generative model
# max => discriminator, discriminative model

# Z ~ P(z)
# the D(G(z)) is important, D=Discriminator, G=Generator

# the discriminator distinguishes between true and fake images



# GANs are generative models, they create new data
# generative models compute joint distributions and can create new data

# Generator: min_{\theta_g} (E{Z~P(Z)} {log(1-D(G(Z)))})

# E{log(D(Z))}
# E{.} = mean = average over a large sample, we sample distributions

# two NNs battle each other until they become experts in their own tasks

# GANs are usually used with images
# GANs can synthesize new images

# do GANs find a Nash equilibrium?
# do GANs find a Nash equilibrium in game theory between two opposing tasks?

# we now import libraries

# Pytorch
import torch

# we use Variable
from torch.autograd import Variable

# torchvision
import torchvision

# we use torchvision
from torchvision import transforms, datasets

# use nn.functional
import torch.nn.functional as F

# use matplotlib
import matplotlib.pyplot as plt

batch_size = 100

train_data = datasets.FashionMNIST(root='fashiondata/',
                                   transform=transforms.ToTensor(),
                                   train=True,
                                   download=True)

test_data = datasets.FashionMNIST(root='fashiondata/',
                                  transform=transforms.ToTensor(),
                                  train=False,
                                  download=True)

# print(train_data[0])
# print(train_data[0][0])

# plt.imshow(train_data[5][0][0], cmap='gray_r')


train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size,
                                            shuffle=True)

test_samples = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size)

# discriminator D
class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # images go from 1x28x28 to 64x14x14
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)

        # images go from 64x14x14 to 128x7x7
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.dense1 = torch.nn.Linear(128 * 7 * 7, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x))).view(-1, 128 * 7 * 7)

        x = F.sigmoid(self.dense1(x))

        return x

# generator G
class generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1 = torch.nn.Linear(128, 256)

        self.dense2 = torch.nn.Linear(256, 1024)

        self.dense3 = torch.nn.Linear(1024, 128 * 7 * 7)

        # from 128x7x7 to 64x14x14
        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # from 64x14x14 to 1x28x28
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

	# we use "BatchNorm1d(.)"
        self.bn1 = torch.nn.BatchNorm1d(256)

	# we use "BatchNorm1d(.)"
        self.bn2 = torch.nn.BatchNorm1d(1024)

	# we use "BatchNorm1d(.)"
        self.bn3 = torch.nn.BatchNorm1d(128 * 7 * 7)

	# we use "BatchNorm2d(.)"
        self.bn4 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))

        x = F.relu(self.bn2(self.dense2(x)))

        x = F.relu(self.bn3(self.dense3(x))).view(-1, 128, 7, 7)

        x = F.relu(self.bn4(self.uconv1(x)))

        s = F.sigmoid(self.uconv(2))

# we use GPU ".cuda()"
d = discriminator().cuda()

# use cuda
g = generator().cuda()

# we define the hyper-parameters
no_epochs = 100

# learning rate
dlr = 0.0003
glr = 0.0003



# we use ADAM momentum
d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)

# we use ADAM momentum
g_optimizer = torch.optim.Adam(g.parameters(), lr=glr)



# training loop
for epoch in range(no_epochs):
    epochdcost = 0
    epochgcost = 0

    for k, (real_images,) in enumerate(train_samples):
        real_images = Variable(real_images).cuda()

        # Z is the noise input
        noise_input = Variable(torch.randn(batch_size, 128)).cuda()

        generated_images = g.forward(noise_input)

        # D(G(Z))
        gen_pred = d.forward(generated_images)

        real_pred = d.forward(real_images)

        # we use the binary cross-entropy cost
        dcost = -torch.sum(torch.log(real_pred) + torch.log(1 - gen_pred)) / batch_size

        gcost = -torch.sum(torch.log(gen_pred)) / batch_size

        d_optimizer.zero_grad()

        # we re-use the graph and we use "retain_graph"
        dcost.backward(retain_graph=True)

        d_optimizer.step()

        g_optimizer.zero_grad()

        gcost.backward()

        g_optimizer.step()

        epochdcost += dcost.data[0]

        epochgcost += gcost.data[0]

    epochcost /= 60000 / batch_size

    epochgcost /= 60000 / batch_size



    print('Epoch: ', epoch)

    print('Discriminator Cost: ', epochdcost)
    print('Generator Cost: ', epochgcost)

    dcosts.append(epochdcost)

    gcosts.append(epochgcost)



    # plot figure with the cost
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_xlabel('Epoch')

    ax.set_ylabel('Cost')

    ax.set_xlim(0, no_epochs)

    ax.plot(dcosts, 'b')



from __future__ import absolute_import
from __future__ import print_function

# numpy
import numpy

#CIFAR-10 Dataset
#CIFAR-100 Dataset
#Caltech-101 Dataset

# we use Sequential
from keras.models import Sequential
from keras.layers import Dense

# use dropout
from keras.layers import Dropout
from keras.layers import Flatten

from keras.constraints import maxnorm
from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
from keras import initializers

from keras import backend as K

K.set_image_dim_ordering('th')

import os
import numpy as np

import scipy.io
import scipy.misc

# matplotlib
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf

def imread(path):
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2:
		img = np.transpose(np.array([img, img, img]), (2, 0, 1))
	return img

#cwd = os.getcwd()
#path = cwd + "/101_ObjectCategories"

#path = "/101_ObjectCategories"
path = "/Users/dionelisnikolaos/Downloads/101_ObjectCategories"

#CIFAR-10 Dataset
#Caltech-101 Dataset

#CIFAR-10 Dataset
#CIFAR-100 Dataset
#Caltech-101 Dataset

valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)

imgs = []
labels = []

print('')

#print(categories)
print(categories[1:])

print('')
categories = categories[1:]

# LOAD ALL IMAGES
for i, category in enumerate(categories):
	iter = 0

	for f in os.listdir(path + "/" + category):
		if iter == 0:
			ext = os.path.splitext(f)[1]

			if ext.lower() not in valid_exts:
				continue

			fullpath = os.path.join(path + "/" + category, f)

			img = scipy.misc.imresize(imread(fullpath), [128, 128, 3])
			img = img.astype('float32')

			img[:, :, 0] -= 123.68
			img[:, :, 1] -= 116.78
			img[:, :, 2] -= 103.94

			imgs.append(img)  # NORMALIZE IMAGE

			label_curr = i
			labels.append(label_curr)

	# iter = (iter+1)%10;

print("Num imgs: %d" % (len(imgs)))
print("Num labels: %d" % (len(labels)))

print(ncategories)

seed = 7
np.random.seed(seed)

# use pandas
import pandas as pd

# use sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.1)

X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)

X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)

print("Num train_imgs: %d" % (len(X_train)))
print("Num test_imgs: %d" % (len(X_test)))

# # one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

print(y_test.shape)

print(X_train[1, 1, 1, :])
print(y_train[1])

# normalize inputs from 0-255 to 0.0-1.0
print(X_train.shape)
print(X_test.shape)

X_train = X_train.transpose(0, 3, 1, 2)
X_test = X_test.transpose(0, 3, 1, 2)

print(X_train.shape)
print(X_test.shape)

# we use scipy
import scipy.io as sio

data = {}

data['categories'] = categories

data['X_train'] = X_train
data['y_train'] = y_train

data['X_test'] = X_test
data['y_test'] = y_test

sio.savemat('caltech_del.mat', data)



# CIFAR-10 Dataset
# CNN model for CIFAR-10

# numpy
import numpy

# CIFAR-10 dataaset
from keras.datasets import cifar10

# Sequential
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Dropout

from keras.layers import Flatten
from keras.constraints import maxnorm

from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

# we now create the model
# we use: https://github.com/acht7111020/CNN_object_classification

# use Sequential
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))

model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 50
lrate = 0.01

decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print('')
print(model.summary())
print('')

# https://github.com/acht7111020/CNN_object_classification
# use: https://github.com/acht7111020/CNN_object_classification



#CIFAR-10 Dataset
#CIFAR-100 Dataset
#Caltech-101 Dataset

# we use Sequential
from keras.models import Sequential
from keras.layers import Dense

# use dropout
from keras.layers import Dropout
from keras.layers import Flatten

from keras.constraints import maxnorm
from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
from keras import initializers

from keras import backend as K

K.set_image_dim_ordering('th')

import os
import numpy as np

import scipy.io
import scipy.misc

# matplotlib
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf

def imread(path):
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2:
		img = np.transpose(np.array([img, img, img]), (2, 0, 1))
	return img

#cwd = os.getcwd()
#path = cwd + "/101_ObjectCategories"

#path = "/101_ObjectCategories"
path = "/Users/dionelisnikolaos/Downloads/101_ObjectCategories"

#CIFAR-10 Dataset
#Caltech-101 Dataset

#CIFAR-10 Dataset
#CIFAR-100 Dataset
#Caltech-101 Dataset

valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)

imgs = []
labels = []

print('')

#print(categories)
print(categories[1:])

print('')
categories = categories[1:]

# LOAD ALL IMAGES
for i, category in enumerate(categories):
	iter = 0

	for f in os.listdir(path + "/" + category):
		if iter == 0:
			ext = os.path.splitext(f)[1]

			if ext.lower() not in valid_exts:
				continue

			fullpath = os.path.join(path + "/" + category, f)

			img = scipy.misc.imresize(imread(fullpath), [128, 128, 3])
			img = img.astype('float32')

			img[:, :, 0] -= 123.68
			img[:, :, 1] -= 116.78
			img[:, :, 2] -= 103.94

			imgs.append(img)  # NORMALIZE IMAGE

			label_curr = i
			labels.append(label_curr)

	# iter = (iter+1)%10;

print("Num imgs: %d" % (len(imgs)))
print("Num labels: %d" % (len(labels)))

print(ncategories)

seed = 7
np.random.seed(seed)

# use pandas
import pandas as pd

# use sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.1)

X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)

X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)

print("Num train_imgs: %d" % (len(X_train)))
print("Num test_imgs: %d" % (len(X_test)))

# # one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

print(y_test.shape)

print(X_train[1, 1, 1, :])
print(y_train[1])

# normalize inputs from 0-255 to 0.0-1.0
print(X_train.shape)
print(X_test.shape)

X_train = X_train.transpose(0, 3, 1, 2)
X_test = X_test.transpose(0, 3, 1, 2)

print(X_train.shape)
print(X_test.shape)

# we use scipy
import scipy.io as sio

data = {}

data['categories'] = categories

data['X_train'] = X_train
data['y_train'] = y_train

data['X_test'] = X_test
data['y_test'] = y_test

sio.savemat('caltech_del.mat', data)



from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# Create the model
model = Sequential()

# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=(3, 128, 128), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile mode
epochs = 300
lrate = 0.0001

decay = lrate / epochs

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = SGD(lr=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

np.random.seed(seed)

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
				 epochs=epochs, batch_size=56, shuffle=True, callbacks=[earlyStopping])

# hist = model.load_weights('./64.15/model.h5');

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1] * 100))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.legend(['train', 'test'])

plt.title('loss')

plt.savefig("loss7.png", dpi=300, format="png")

plt.figure()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.legend(['train', 'test'])

plt.title('accuracy')

plt.savefig("accuracy7.png", dpi=300, format="png")

model_json = model.to_json()

with open("model7.json", "w") as json_file:
	json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model7.h5")
print("Saved model to disk")



#import numpy
import numpy as np

# https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2
# use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2

# we use: https://skymind.ai/wiki/open-datasets
# use: http://people.csail.mit.edu/yalesong/cvpr12/

from keras.datasets import mnist
((trainX, trainY), (testX, testY)) = mnist.load_data()

print(trainX.shape)
print(testX.shape)

from keras.datasets import fashion_mnist
((trainX2, trainY2), (testX2, testY2)) = fashion_mnist.load_data()

print(trainX2.shape)
print(testX2.shape)

print('')

from keras.datasets import imdb
((trainX3, trainY3), (testX3, testY3)) = imdb.load_data()

print(trainX3.shape)
print(testX3.shape)

print('')

from keras.datasets import cifar10
((trainX4, trainY4), (testX4, testY4)) = cifar10.load_data()

print(trainX4.shape)
print(testX4.shape)

from keras.datasets import cifar100
((trainX5, trainY5), (testX5, testY5)) = cifar100.load_data()

print(trainX5.shape)
print(testX5.shape)

print('')

# use: https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d
# we now use: https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d

from keras.datasets import reuters
((trainX6, trainY6), (testX6, testY6)) = reuters.load_data()

print(trainX6.shape)
print(testX6.shape)

from keras.datasets import boston_housing
((trainX7, trainY7), (testX7, testY7)) = boston_housing.load_data()

print(trainX7.shape)
print(testX7.shape)

print('')

# use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2
# we use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#from sklearn import datasets
#from sklearn import datasets2

import sklearn
#from sklearn.datasets2 import kddcup99

#import sklearn.datasets2
#import sklearn.datasets

#dataset_boston = datasets.load_boston()
#dataset_boston = datasets2.load_boston()

#dataset_kddcup99 = datasets2.load_digits()



# use .io
import scipy.io

#mat2 = scipy.io.loadmat('NATOPS6.mat')
mat2 = scipy.io.loadmat('/Users/dionelisnikolaos/Downloads/NATOPS6.mat')

# NATOPS6.mat
print(mat2)

#mat = scipy.io.loadmat('thyroid.mat')
mat = scipy.io.loadmat('/Users/dionelisnikolaos/Downloads/thyroid.mat')

# thyroid.mat
print(mat)



# usenet_recurrent3.3.data
# we use: usenet_recurrent3.3.data

# use pandas
import pandas as pd

# numpy
import numpy

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss

from sklearn.model_selection import train_test_split

data_dir = "/Users/dionelisnikolaos/Downloads/"
raw_data_filename = data_dir + "usenet_recurrent3.3.data"

#raw_data_filename = "/Users/dionelisnikolaos/Downloads/usenet_recurrent3.3.data"

# raw_data_filename = "/Users/dionelisnikolaos/Downloads/usenet_recurrent3.3.data"
# use: raw_data_filename = "/Users/dionelisnikolaos/Downloads/usenet_recurrent3.3.data"

print ("Loading raw data")

raw_data = pd.read_csv(raw_data_filename, header=None)

print ("Transforming data")

# Categorize columns: "protocol", "service", "flag", "attack_type"
raw_data[1], protocols= pd.factorize(raw_data[1])
raw_data[2], services = pd.factorize(raw_data[2])

raw_data[3], flags    = pd.factorize(raw_data[3])
raw_data[41], attacks = pd.factorize(raw_data[41])

# separate features (columns 1..40) and label (column 41)
features= raw_data.iloc[:,:raw_data.shape[1]-1]
labels= raw_data.iloc[:,raw_data.shape[1]-1:]

# convert them into numpy arrays
#features= numpy.array(features)

#labels= numpy.array(labels).ravel() # this becomes an 'horizontal' array
labels= labels.values.ravel() # this becomes a 'horizontal' array

# Separate data in train set and test set
df= pd.DataFrame(features)

# create training and testing vars
# Note: train_size + test_size < 1.0 means we are subsampling

# Use small numbers for slow classifiers, as KNN, Radius, SVC,...
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2)

print('')

print ("X_train, y_train:", X_train.shape, y_train.shape)
print ("X_test, y_test:", X_test.shape, y_test.shape)

print('')

print(X_train.shape)
print(y_train.shape)

print('')

print(X_train.shape)
print(X_test.shape)

print('')



# use matplotlib
import matplotlib.pyplot as plt

# we use: https://skymind.ai/wiki/open-datasets
# use: http://people.csail.mit.edu/yalesong/cvpr12/

from csv import reader

# Load a CSV file
def load_csv(filename):
    file = open(filename, "r")

    lines = reader(file)
    dataset = list(lines)

    return dataset

dataset = load_csv('/Users/dionelisnikolaos/Downloads/ann-train.data.txt')

# Load dataset

filename = '/Users/dionelisnikolaos/Downloads/ann-train.data.txt'
#print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))

#file = open(filename, 'r')
#for line in file:
#    print (line,)

text_file = open(filename, "r")
lines = text_file.read().split(' ')

#print(lines)

list_of_lists = []

with open(filename) as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]

        # in alternative, if you need to use the file content as numbers
        # inner_list = [int(elt.strip()) for elt in line.split(',')]
        list_of_lists.append(inner_list)

print(list_of_lists)



import pandas as pd
import numpy

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss

from sklearn.model_selection import train_test_split

#data_dir="./datasets/KDD-CUP-99/"
#data_dir="./"

data_dir = "/Users/dionelisnikolaos/Downloads/"
raw_data_filename = data_dir + "kddcup.data"

#raw_data_filename = "/Users/dionelisnikolaos/Downloads/kddcup.data"

print ("Loading raw data")

raw_data = pd.read_csv(raw_data_filename, header=None)

print ("Transforming data")

# Categorize columns: "protocol", "service", "flag", "attack_type"
raw_data[1], protocols= pd.factorize(raw_data[1])
raw_data[2], services = pd.factorize(raw_data[2])

raw_data[3], flags    = pd.factorize(raw_data[3])
raw_data[41], attacks = pd.factorize(raw_data[41])

# separate features (columns 1..40) and label (column 41)
features= raw_data.iloc[:,:raw_data.shape[1]-1]
labels= raw_data.iloc[:,raw_data.shape[1]-1:]

# convert them into numpy arrays
#features= numpy.array(features)

# this becomes an 'horizontal' array
#labels= numpy.array(labels).ravel()

# this becomes a 'horizontal' array
labels= labels.values.ravel()

# Separate data in train set and test set
df= pd.DataFrame(features)

# create training and testing vars
# Note: train_size + test_size < 1.0 means we are subsampling

# Use small numbers for slow classifiers, as KNN, Radius, SVC,...
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2)

print('')

print ("X_train, y_train:", X_train.shape, y_train.shape)
print ("X_test, y_test:", X_test.shape, y_test.shape)

print('')

print(X_train.shape)
print(y_train.shape)

print('')

print(X_train.shape)
print(X_test.shape)

print('')

# Training, choose model by commenting/uncommenting clf=
print ("Training model")

clf= RandomForestClassifier(n_jobs=-1, random_state=3, n_estimators=102)#, max_features=0.8, min_samples_leaf=3, n_estimators=500, min_samples_split=3, random_state=10, verbose=1)
#clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, presort=False)

trained_model= clf.fit(X_train, y_train)

print ("Score: ", trained_model.score(X_train, y_train))

# Predicting
print ("Predicting")

y_pred = clf.predict(X_test)

print ("Computing performance metrics")

results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)

print ("Confusion matrix:\n", results)
print ("Error: ", error)

# KDD99 Dataset
# use: https://github.com/ghuecas/kdd99ml

# https://github.com/ghuecas/kdd99ml
# we use: https://github.com/ghuecas/kdd99ml



import json
import datetime

import os
import numpy as np

# make keras deterministic
#np.random.seed(42)

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.callbacks import CallbackList, ModelCheckpoint
from keras.regularizers import l2

import os

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#from keras.applications.inception_v3 import InceptionV3
#base_model = InceptionV3(weights='imagenet', include_top=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

num_train_images =  1500
num_test_images = 100

#-------------------
# organize imports
#-------------------
import numpy as np

import os
import h5py

import glob
import cv2

# we use opencv-python
import cv2

# we use keras
from keras.preprocessing import image

#------------------------
# dataset pre-processing
#------------------------
#train_path   = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\train"
#test_path    = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test"

#train_path   = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\train"
train_path   = "/Users/dionelisnikolaos/Downloads/dataset/train"

#test_path    = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test"
test_path    = "/Users/dionelisnikolaos/Downloads/dataset/test"

train_labels = os.listdir(train_path)
test_labels  = os.listdir(test_path)

# tunable parameters
image_size       = (64, 64)

num_train_images = 1500
num_test_images  = 100

num_channels     = 3

# train_x dimension = {(64*64*3), 1500}
# train_y dimension = {1, 1500}
# test_x dimension  = {(64*64*3), 100}
# test_y dimension  = {1, 100}

train_x = np.zeros(((image_size[0]*image_size[1]*num_channels), num_train_images))
train_y = np.zeros((1, num_train_images))

test_x  = np.zeros(((image_size[0]*image_size[1]*num_channels), num_test_images))
test_y  = np.zeros((1, num_test_images))

#----------------
# TRAIN dataset
#----------------
count = 0
num_label = 0

for i, label in enumerate(train_labels):
	cur_path = train_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x   = image.img_to_array(img)
		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)

		train_x[:,count] = x
		train_y[:,count] = num_label

		count += 1
	num_label += 1

#--------------
# TEST dataset
#--------------
count = 0
num_label = 0

for i, label in enumerate(test_labels):
	cur_path = test_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x   = image.img_to_array(img)
		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)

		test_x[:,count] = x
		test_y[:,count] = num_label

		count += 1
	num_label += 1

#------------------
# standardization
#------------------
train_x = train_x/255.
test_x  = test_x/255.

print ("train_labels : " + str(train_labels))

print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))

print ("test_x shape : " + str(test_x.shape))
print ("test_y shape : " + str(test_y.shape))

print('')

# train_x and test_x
print(train_x.shape)
print(test_x.shape)

# https://gogul09.github.io/software/neural-nets-logistic-regression
# use: https://gogul09.github.io/software/neural-nets-logistic-regression

#-----------------
# save using h5py
#-----------------
h5_train = h5py.File("train_x.h5", 'w')
h5_train.create_dataset("data_train", data=np.array(train_x))

h5_train.close()

h5_test = h5py.File("test_x.h5", 'w')
h5_test.create_dataset("data_test", data=np.array(test_x))

h5_test.close()

def sigmoid(z):
	return (1/(1+np.exp(-z)))

def init_params(dimension):
	w = np.zeros((dimension, 1))
	b = 0
	return w, b

def propagate(w, b, X, Y):
	# num of training samples
	m = X.shape[1]

	# forward pass
	A    = sigmoid(np.dot(w.T,X) + b)
	cost = (-1/m)*(np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A))))

	# back propagation
	dw = (1/m)*(np.dot(X, (A-Y).T))
	db = (1/m)*(np.sum(A-Y))

	cost = np.squeeze(cost)

	# gradient dictionary
	grads = {"dw": dw, "db": db}

	return grads, cost

def optimize(w, b, X, Y, epochs, lr):
	costs = []
	for i in range(epochs):
		# calculate gradients
		grads, cost = propagate(w, b, X, Y)

		# get gradients
		dw = grads["dw"]
		db = grads["db"]

		# update rule
		w = w - (lr*dw)
		b = b - (lr*db)

		if i % 100 == 0:
			costs.append(cost)
			print ("cost after %i epochs: %f" %(i, cost))

	# param dict
	params = {"w": w, "b": b}

	# gradient dict
	grads  = {"dw": dw, "db": db}

	return params, grads, costs

def predict(w, b, X):
	m = X.shape[1]

	Y_predict = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(np.dot(w.T, X) + b)

	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_predict[0, i] = 0
		else:
			Y_predict[0,i]  = 1

	return Y_predict

def predict_image(w, b, X):
	Y_predict = None

	w = w.reshape(X.shape[0], 1)
	A = sigmoid(np.dot(w.T, X) + b)

	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_predict = 0
		else:
			Y_predict = 1

	return Y_predict

def model(X_train, Y_train, X_test, Y_test, epochs, lr):
	w, b = init_params(X_train.shape[0])
	params, grads, costs = optimize(w, b, X_train, Y_train, epochs, lr)

	w = params["w"]
	b = params["b"]

	Y_predict_train = predict(w, b, X_train)
	Y_predict_test  = predict(w, b, X_test)

	print ("train_accuracy: {} %".format(100-np.mean(np.abs(Y_predict_train - Y_train)) * 100))
	print ("test_accuracy : {} %".format(100-np.mean(np.abs(Y_predict_test  - Y_test)) * 100))

	log_reg_model = {"costs": costs,
				     "Y_predict_test": Y_predict_test,
					 "Y_predict_train" : Y_predict_train,
					 "w" : w,
					 "b" : b,
					 "learning_rate" : lr,
					 "epochs": epochs}

	return log_reg_model

# we use: https://gogul09.github.io/software/neural-nets-logistic-regression

#epochs = 100
epochs = 10

# lr, learning rate, step size
lr = 0.0003

# activate the logistic regression model
myModel = model(train_x, train_y, test_x, test_y, epochs, lr)

#test_img_paths = ["G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0723.jpg",
#                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0713.jpg",
#                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0782.jpg",
#                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0799.jpg",
#                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\test_1.jpg"]

# https://gogul09.github.io/software/neural-nets-logistic-regression
# use: https://gogul09.github.io/software/neural-nets-logistic-regression

test_img_paths = ["/Users/dionelisnikolaos/Downloads/dataset/test/airplane/image_0763.jpg",
                  "/Users/dionelisnikolaos/Downloads/dataset/test/airplane/image_0753.jpg",
                  "/Users/dionelisnikolaos/Downloads/dataset/test/bike/image_0782.jpg",
                  "/Users/dionelisnikolaos/Downloads/dataset/test/bike/image_0799.jpg",
                  "/Users/dionelisnikolaos/Downloads/dataset/test/bike/image_0751.jpg"]

for test_img_path in test_img_paths:
	img_to_show    = cv2.imread(test_img_path, -1)
	img            = image.load_img(test_img_path, target_size=image_size)
	x              = image.img_to_array(img)
	x              = x.flatten()
	x              = np.expand_dims(x, axis=1)
	predict        = predict_image(myModel["w"], myModel["b"], x)
	predict_label  = ""

	if predict == 0:
		predict_label = "airplane"
	else:
		predict_label = "bike"

	# display the test image and the predicted label
	cv2.putText(img_to_show, predict_label, (30,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("test_image", img_to_show)

	key = cv2.waitKey(0) & 0xFF

	if (key == 27):
		cv2.destroyAllWindows()



import keras
import keras.datasets

# use datasets
import keras.datasets

from keras.datasets import cifar10
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()



from keras.datasets import fashion_mnist
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# set the matplotlib backend so figures can be saved in the background
import matplotlib

#matplotlib.use("Agg")

# import the necessary packages
from sklearn.metrics import classification_report
from keras.optimizers import SGD

# use Fashion-MNIST
from keras.datasets import fashion_mnist

from keras.utils import np_utils
from keras import backend as K

#from imutils import build_montages
import numpy as np

# use matplotlib
import matplotlib.pyplot as plt

#image_index = 7777
image_index = 777

# ((trainX, trainY), (testX, testY))
# (x_train, y_train), (x_test, y_test)
y_train = trainY
x_train = trainX

# ((trainX, trainY), (testX, testY))
# (x_train, y_train), (x_test, y_test)
y_test = testY
x_test = testX

print(trainX.shape)
print(trainY.shape)

print(testX.shape)
print(testY.shape)

print(y_train[image_index].shape)
print(x_train[image_index].shape)

print(y_train[image_index])

plt.imshow(x_train[image_index], cmap='Greys')
#plt.imshow(x_train[image_index])

#plt.pause(5)
plt.pause(2)

#x_train.shape
print(x_train.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# we define the input shape
input_shape = (28, 28, 1)

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

# import the necessary packages
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# import the necessary packages
from keras.layers.core import Activation
from keras.layers.core import Flatten

# use dropout
from keras.layers.core import Dropout

from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()

        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

            chanDim = 1

            # first CONV => RELU => CONV => RELU => POOL layer set
            model.add(Conv2D(32, (3, 3), padding="same",
                             input_shape=inputShape))

            model.add(Activation("relu"))

            model.add(BatchNormalization(axis=chanDim))
            model.add(Conv2D(32, (3, 3), padding="same"))

            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # second CONV => RELU => CONV => RELU => POOL layer set
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation("relu"))

            model.add(BatchNormalization(axis=chanDim))
            model.add(Conv2D(64, (3, 3), padding="same"))

            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # first (and only) set of FC => RELU layers
            model.add(Flatten())
            model.add(Dense(512))

            model.add(Activation("relu"))
            model.add(BatchNormalization())

            model.add(Dropout(0.5))

            # softmax classifier
            model.add(Dense(classes))
            model.add(Activation("softmax"))

            # return the constructed network architecture
            return model



# use numpy
import numpy as np

#matplotlib inline
import matplotlib.pyplot as plt

# use tensorflow
import tensorflow as tf

# we use the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
# use: https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

# use matplotlib
import matplotlib.pyplot as plt

image_index = 7777

# The label is 8
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')

#plt.pause(5)
plt.pause(2)

#x_train.shape
print(x_train.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# we define the input shape
input_shape = (28, 28, 1)

# the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)

print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])



# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 2D arrays for fully connected layers
model.add(Flatten())

model.add(Dense(128, activation=tf.nn.relu))

model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ADAM, adaptive momentum
# we use the Adam optimizer

# fit the model
#model.fit(x=x_train,y=y_train, epochs=10)

#model.fit(x=x_train,y=y_train, epochs=10)
model.fit(x=x_train,y=y_train, epochs=8)

# evaluate the model
model.evaluate(x_test, y_test)

# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

# use index 4444
image_index = 4444

plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')

#plt.pause(5)
plt.pause(2)

#pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))

print(pred.argmax())



# Deep Generative Models
# GANs and VAEs, Generative Models

# random noise
# from random noise to a tensor

# We use batch normalisation.
# GANs are very difficult to train. Super-deep models. This is why we use batch normalisation.

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb



# Anomaly detection (AD)
# Unsupervised machine learning

# GANs for super-resolution
# Generative Adversarial Networks, GANs

# the BigGAN dataset
# BigGAN => massive dataset
# latent space, BigGAN, GANs

# down-sampling, sub-sample, pooling
# throw away samples, pooling, max-pooling

# partial derivatives
# loss function and partial derivatives

# https://github.com/Students-for-AI/The-Academy-of-AI
# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models

# Generator G and Discriminator D
# the loss function of the Generator G

# up-convolution
# We use a filter we do up-convolution with.

# use batch normalisation
# GANs are very difficult to train and this is why we use batch normalisation.

# We normalize across a batch.
# Mean across a batch. We use batches. Normalize across a batch.

# the ReLU activation function
# ReLU is the most common activation function. We use ReLU.

# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb



# we use PyTorch
import torch

#import torch
import torchvision

from torchvision import datasets, transforms

# use matplotlib
import matplotlib.pyplot as plt

#import torch
#import torchvision

#from torchvision import transforms, datasets

# use nn.functional
import torch.nn.functional as F

#import matplotlib.pyplot as plt
#batch_size = 128

# download the training dataset
#train_data = datasets.FashionMNIST(root='fashiondata/',
#                                   transform=transforms.ToTensor(),
#                                   train=True,
#                                   download=True)

# we create the train data loader
#train_loader = torch.utils.data.DataLoader(train_data,
#                                           shuffle=True,
#                                           batch_size=batch_size)

# define the batch size
batch_size = 100

train_data = datasets.FashionMNIST(root='fashiondata/',
                                 transform=transforms.ToTensor(),
                                 train=True,
                                 download=True
                                 )

train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True
                                           )

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# class for D and G
# we train the discriminator and the generator

# we make the discriminator
class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)  # 1x28x28-> 64x14x14
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 64x14x14-> 128x7x7

        self.dense1 = torch.nn.Linear(128 * 7 * 7, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))).view(-1, 128 * 7 * 7)

        # use sigmoid for the output layer
        x = F.sigmoid(self.dense1(x))

        return x

# this was for the discriminator
# we now do the same for the generator

# Generator G
class generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(256, 1024)
        self.dense3 = torch.nn.Linear(1024, 128 * 7 * 7)

        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 128x7x7 -> 64x14x14
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)  # 64x14x14 -> 1x28x28

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(1024)

        self.bn3 = torch.nn.BatchNorm1d(128 * 7 * 7)
        self.bn4 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))

        x = F.relu(self.bn3(self.dense3(x))).view(-1, 128, 7, 7)

        x = F.relu(self.bn4(self.uconv1(x)))

        x = F.sigmoid(self.uconv2(x))
        return x

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb
# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# instantiate the model
d = discriminator()
g = generator()

# training hyperparameters
#epochs = 100

#epochs = 100
epochs = 10

# learning rate
#dlr = 0.0003
#glr = 0.0003

dlr = 0.003
glr = 0.003

d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)
g_optimizer = torch.optim.Adam(g.parameters(), lr=glr)

dcosts = []
gcosts = []

plt.ion()
fig = plt.figure()

loss_ax = fig.add_subplot(121)
loss_ax.set_xlabel('Batch')

loss_ax.set_ylabel('Cost')
loss_ax.set_ylim(0, 0.2)

generated_img = fig.add_subplot(122)

plt.show()

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

def train(epochs, glr, dlr):
    g_losses = []
    d_losses = []

    for epoch in range(epochs):

        # iteratre over mini-batches
        for batch_idx, (real_images, _) in enumerate(train_samples):

            z = torch.randn(batch_size, 128)  # generate random latent variable to generate images from
            generated_images = g.forward(z)  # generate images

            gen_pred = d.forward(generated_images)  # prediction of discriminator on generated batch
            real_pred = d.forward(real_images)  # prediction of discriminator on real batch

            dcost = -torch.sum(torch.log(real_pred)) - torch.sum(torch.log(1 - gen_pred))  # cost of discriminator
            gcost = -torch.sum(torch.log(gen_pred)) / batch_size  # cost of generator

            # train discriminator
            d_optimizer.zero_grad()
            dcost.backward(retain_graph=True)  # retain the computational graph so we can train generator after
            d_optimizer.step()

            # train generator
            g_optimizer.zero_grad()

            gcost.backward()
            g_optimizer.step()

            # give us an example of a generated image after every 10000 images produced
            #if batch_idx * batch_size % 10000 == 0:

            # give us an example of a generated image after every 20 images produced
            if batch_idx % 20 == 0:
                g.eval()  # put in evaluation mode
                noise_input = torch.randn(1, 128)
                generated_image = g.forward(noise_input)

                generated_img.imshow(generated_image.detach().squeeze(), cmap='gray_r')

                # pause for some seconds
                plt.pause(5)

                # put back into training mode
                g.train()

            dcost /= batch_size
            gcost /= batch_size

            print('Epoch: ', epoch, 'Batch idx:', batch_idx, '\tDisciminator cost: ', dcost.item(),
                  '\tGenerator cost: ', gcost.item())

            dcosts.append(dcost)
            gcosts.append(gcost)

            loss_ax.plot(dcosts, 'b')
            loss_ax.plot(gcosts, 'r')

            fig.canvas.draw()

#print(torch.__version__)
train(epochs, glr, dlr)

# We obtain:
# Epoch:  0 Batch idx: 0 	Disciminator cost:  1.3832124471664429 	Generator cost:  0.006555716972798109
# Epoch:  0 Batch idx: 1 	Disciminator cost:  1.0811840295791626 	Generator cost:  0.008780254982411861
# Epoch:  0 Batch idx: 2 	Disciminator cost:  0.8481155633926392 	Generator cost:  0.011281056329607964
# Epoch:  0 Batch idx: 3 	Disciminator cost:  0.6556042432785034 	Generator cost:  0.013879001140594482
# Epoch:  0 Batch idx: 4 	Disciminator cost:  0.5069876909255981 	Generator cost:  0.016225570812821388
# Epoch:  0 Batch idx: 5 	Disciminator cost:  0.4130948781967163 	Generator cost:  0.018286770209670067
# Epoch:  0 Batch idx: 6 	Disciminator cost:  0.33445805311203003 	Generator cost:  0.02015063539147377
# Epoch:  0 Batch idx: 7 	Disciminator cost:  0.279323011636734 	Generator cost:  0.021849267184734344
# Epoch:  0 Batch idx: 8 	Disciminator cost:  0.2245958000421524 	Generator cost:  0.02352861315011978
# Epoch:  0 Batch idx: 9 	Disciminator cost:  0.18664218485355377 	Generator cost:  0.025215130299329758
# Epoch:  0 Batch idx: 10 	Disciminator cost:  0.14700829982757568 	Generator cost:  0.02692217379808426

# Epoch:  0 Batch idx: 34 	Disciminator cost:  0.31759023666381836 	Generator cost:  0.02075548656284809
# Epoch:  0 Batch idx: 35 	Disciminator cost:  0.35554683208465576 	Generator cost:  0.018939709290862083
# Epoch:  0 Batch idx: 36 	Disciminator cost:  0.07700302451848984 	Generator cost:  0.04144695773720741
# Epoch:  0 Batch idx: 37 	Disciminator cost:  0.08900360018014908 	Generator cost:  0.05888563022017479
# Epoch:  0 Batch idx: 38 	Disciminator cost:  0.0921328067779541 	Generator cost:  0.0593753345310688
# Epoch:  0 Batch idx: 39 	Disciminator cost:  0.09943853318691254 	Generator cost:  0.05279992148280144
# Epoch:  0 Batch idx: 40 	Disciminator cost:  0.2455407679080963 	Generator cost:  0.036564696580171585
# Epoch:  0 Batch idx: 41 	Disciminator cost:  0.10074597597122192 	Generator cost:  0.03721988573670387
# Epoch:  0 Batch idx: 42 	Disciminator cost:  0.07906078547239304 	Generator cost:  0.04363853484392166

# Epoch:  0 Batch idx: 111 	Disciminator cost:  0.49694764614105225 	Generator cost:  0.02403220161795616
# Epoch:  0 Batch idx: 112 	Disciminator cost:  0.581132173538208 	Generator cost:  0.026757290586829185
# Epoch:  0 Batch idx: 113 	Disciminator cost:  0.16659873723983765 	Generator cost:  0.0335114412009716
# Epoch:  0 Batch idx: 114 	Disciminator cost:  0.0639999508857727 	Generator cost:  0.04211951419711113
# Epoch:  0 Batch idx: 115 	Disciminator cost:  0.018385086208581924 	Generator cost:  0.05511172115802765
# Epoch:  0 Batch idx: 116 	Disciminator cost:  0.012170110829174519 	Generator cost:  0.06555930525064468
# Epoch:  0 Batch idx: 117 	Disciminator cost:  0.006641524378210306 	Generator cost:  0.07086272537708282
# Epoch:  0 Batch idx: 118 	Disciminator cost:  0.010556117631494999 	Generator cost:  0.06929603219032288
# Epoch:  0 Batch idx: 119 	Disciminator cost:  0.017774969339370728 	Generator cost:  0.07270769774913788

# Epoch:  0 Batch idx: 444 	Disciminator cost:  0.06787727028131485 	Generator cost:  0.04046594724059105
# Epoch:  0 Batch idx: 445 	Disciminator cost:  0.07139576226472855 	Generator cost:  0.03837932273745537
# Epoch:  0 Batch idx: 446 	Disciminator cost:  0.08202749490737915 	Generator cost:  0.039551254361867905
# Epoch:  0 Batch idx: 447 	Disciminator cost:  0.12328958511352539 	Generator cost:  0.03817861154675484
# Epoch:  0 Batch idx: 448 	Disciminator cost:  0.06865841150283813 	Generator cost:  0.03938257694244385

# generate random latent variable to generate images
z = torch.randn(batch_size, 128)

# generate images
im = g.forward(z)
# use "forward(.)"

plt.imshow(im)

