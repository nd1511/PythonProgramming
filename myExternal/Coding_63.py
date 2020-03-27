from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
print(tf.__version__) # 1.14.0

# https://drive.google.com/drive/folders/1DY-hs7-qUprvGgkSX79QsyuUujSnfFCj?usp=sharing
# use: https://drive.google.com/drive/folders/1DY-hs7-qUprvGgkSX79QsyuUujSnfFCj?usp=sharing

# https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2
# https://www.analyticsinsight.net/best-computer-vision-courses-to-master-in-2019/
# UCI data: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
# Human Activity Recognition Using Smartphones Data Set, archive.ics.uci.edu, Human Activity Recognition

import sys
import numpy
import os, tarfile, errno
import matplotlib.pyplot as plt

import sklearn
import numpy.random
import scipy.stats as ss

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib
else:
    import urllib

try:
    from imageio import imsave
except:
    from scipy.misc import imsave

print(sys.version_info) # we use: sys.version_info
from sklearn.ensemble import IsolationForest # Import IsolationForest module
# use: https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1

# https://bhtradingchallenge.com
# use: https://bhtradingchallenge.com

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import cv2
import numpy.random
import scipy.stats as ss
from sklearn.mixture import GaussianMixture

import tensorflow as tf
from sklearn import metrics

# use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2
# https://www.analyticsinsight.net/best-computer-vision-courses-to-master-in-2019/

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy
from scipy import ndimage, misc
from scipy.misc import imshow

from gluoncv import data, utils
from gluoncv.data import ImageNet
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# sklearn.datasets.make_moons(n_samples=100, shuffle=True, noise=None, random_state=None)
# Use: sklearn.datasets.make_moons(n_samples=100, shuffle=True, noise=None, random_state=None)

# Make two interleaving half circles: A toy dataset to visualize clustering and classification algorithms.
# We now use: sklearn.datasets.make_moons(n_samples=100, shuffle=True, noise=None, random_state=None)
# Parameters: n_samples : int, optional (default=100). The total number of points generated.
# shuffle : bool, optional (default=True). Whether to shuffle the samples.
# noise : double or None (default=None). Standard deviation of Gaussian noise added to the data.
# random_state : int, RandomState instance or None (default)
# Determines random number generation for dataset shuffling and noise.
# Returns: X : array of shape [n_samples, 2]. The generated samples.
# y : array of shape [n_samples]. The integer labels (0 or 1) for class membership of each sample.

from sklearn import datasets as dsets
X_moon, y_moon = dsets.make_moons(n_samples=200, shuffle=True, noise=0.09)

print(X_moon.shape)
print(y_moon.shape)

plt.plot(X_moon[:,0], X_moon[:,1], 'o')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('./HalfMoon_dataset.png')
plt.show()

import time
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_moons, make_blobs

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
outliers_fraction = 0.15 # Example settings
n_samples = 300 # Set example settings

n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# define the anomaly detection methods to be compared

# define the anomaly detection methods to be compared
anomaly_algorithms = [("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))]

# define the datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)

datasets = [make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0],
            make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5], **blobs_params)[0],
            make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3], **blobs_params)[0],
            4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] - np.array([0.5, 0.25])),
            14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

# compare the given classifiers under the given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))

plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

plot_num = 1
rng = np.random.RandomState(42)

for i_dataset, X in enumerate(datasets): # Add the outliers
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)

    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        algorithm.fit(X)

        t1 = time.time()
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)

        if i_dataset == 0:
            plt.title(name, size=18)

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)

        if name != "Local Outlier Factor":  # the LOF does not implement predict
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) # plot level lines and points

            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)

        plt.xticks(())
        plt.yticks(())

        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15, horizontalalignment='right')
        plot_num += 1

plt.savefig('./OoD_AnomalyDetection.png')
plt.show()



import numpy as np
import tensorflow as tf

ds = tf.contrib.distributions

# MNIST: Keras or scikit-learn embedded datasets
# For example, Keras: from keras.datasets import mnist

#def sample_mog(batch_size, n_mixture=8, std=0.01, radius=1.0):
def sample_mog(batch_size, n_mixture=6, std=0.03, radius=1.0):
    #thetas = np.linspace(0, 2 * np.pi, n_mixture)

    thetas = np.linspace(0, 2 * np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)

    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]

    data = ds.Mixture(cat, comps)
    return data.sample(batch_size)

print(sample_mog(128)) # sample_mog(128)
samplePoints = sample_mog(100)

print(samplePoints)
tf.InteractiveSession()
samplePoints2 = samplePoints.eval()

#plt.plot(samplePoints2[:,0], samplePoints2[:,1])
plt.plot(samplePoints2[:,0], samplePoints2[:,1], 'o')

plt.xlabel('x')
plt.ylabel('y')

plt.savefig('./2Dmixtures.png')
plt.show()

samplePoints = sample_mog(100, 4, 0.03, 0.7)
print(samplePoints)

tf.InteractiveSession()
samplePoints2 = samplePoints.eval()

#plt.plot(samplePoints2[:,0], samplePoints2[:,1])
plt.plot(samplePoints2[:,0], samplePoints2[:,1], 'o')

plt.xlabel('x')
plt.ylabel('y')

plt.savefig('./2Dmixtures2.png')
plt.show()



image_ind = 10 # we define the index
#train_data = sio.loadmat('train_32x32.mat')
train_data = sio.loadmat('/Users/dionelisnikolaos/Downloads/train_32x32.mat')

# The SVHN Dataset
# Street View House Numbers (SVHN)

# we access the dict
x_train = train_data['X']
y_train = train_data['y']

plt.imshow(x_train[:,:,:,image_ind])
plt.show() # we show the sample

print(y_train[image_ind])

image_ind = 10 # index, define the image index
test_data = sio.loadmat('/Users/dionelisnikolaos/Downloads/test_32x32.mat')

x_test = test_data['X'] # access the dict
y_test = test_data['y'] # access to the dict

plt.imshow(x_test[:,:,:,image_ind])
plt.show() # show the sample

print(y_test[image_ind])

# Import Line2D for marking legend in graph
from matplotlib.lines import Line2D

mean = [0, 0] # we define the mean vector
cov = [[1, 0], [0, 100]] # diagonal covariance

import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 1000).T

plt.plot(x, y, 'o')
plt.axis('equal')

plt.xlabel('x')
plt.ylabel('y')

plt.savefig('./MultivariateNormal.png')
plt.show()

x, y = np.random.multivariate_normal([0, 0], [[100, 0], [0, 1]], 1000).T
plt.plot(x, y, 'o')
plt.axis('equal')

plt.xlabel('x')
plt.ylabel('y')

plt.savefig('./MultivariateNormal2.png')
plt.show()



n = 10000
numpy.random.seed(0x5eed)

# Parameters of the mixture components
norm_params = np.array([[5, 1], [1, 1.3], [9, 1.3]])

n_components = norm_params.shape[0] # Components and weights of each component
weights = np.ones(n_components, dtype=np.float64) / float(n_components) # Weight of each component

mixture_idx = numpy.random.choice(n_components, size=n, replace=True, p=weights) # Indices to choose the component
y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx), dtype=np.float64) # y is the mixture sample

xs = np.linspace(y.min(), y.max(), 200) # Theoretical PDF plotting
ys = np.zeros_like(xs) # Generate the x and y plotting positions

for (l, s), w in zip(norm_params, weights):
    ys += ss.norm.pdf(xs, loc=l, scale=s) * w

plt.plot(xs, ys)
plt.hist(y, normed=True, bins="fd")

plt.xlabel("x")
plt.ylabel("f(x)")

plt.show()



# Generate synthetic data
N,D = 1000, 2 # number of points and dimensionality

if D == 2:
    #set gaussian ceters and covariances in 2D

    #set gaussian ceters and covariances in 2D
    means = np.array([[0.5, 0.0], [0, 0], [-0.5, -0.5], [-0.8, 0.3]])

    covs = np.array([np.diag([0.01, 0.01]), np.diag([0.025, 0.01]),
                     np.diag([0.01, 0.025]), np.diag([0.01, 0.01])])

elif D == 3:
    # set gaussian ceters and covariances in 3D

    # set gaussian ceters and covariances in 3D
    means = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [-0.5, -0.5, -0.5], [-0.8, 0.3, 0.4]])

    covs = np.array([np.diag([0.01, 0.01, 0.03]), np.diag([0.08, 0.01, 0.01]),
                     np.diag([0.01, 0.05, 0.01]), np.diag([0.03, 0.07, 0.01])])

n_gaussians = means.shape[0]

points = []
for i in range(len(means)):
    x = np.random.multivariate_normal(means[i], covs[i], N )
    points.append(x)

points = np.concatenate(points)



# Generate a normally distributed data set for training

# Generate a normally distributed data set for training
X = 0.3 * np.random.randn(100, 2)
X_train_normal = np.r_[X + 2, X - 2]

# Generating outliers for training
X_train_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# Generating a normally distributed dataset for testing
X = 0.3 * np.random.randn(20, 2)
X_test_normal = np.r_[X + 2, X - 2]

# Generating outliers for testing
X_test_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

#Plotting and visualising the data points
plt.figure(figsize=(10,7.5))

plt.scatter(X_train_normal[:,0],X_train_normal[:,1],label='X_train_normal')
#plt.scatter(X_train_outliers[:,0],X_train_outliers[:,1],label='X_train_outliers')

plt.scatter(X_test_normal[:,0],X_test_normal[:,1],label='X_test_normal')
#plt.scatter(X_test_outliers[:,0],X_test_outliers[:,1],label='X_test_outliers')
plt.scatter(X_train_outliers[:,0],X_train_outliers[:,1],label='X_test_outliers')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.savefig('./DataNormalAbnormal.png')
plt.show()

#Plotting and visualising the data points
plt.figure(figsize=(10,7.5))

plt.scatter(X_train_normal[:,0],X_train_normal[:,1],label='X_train_normal')
plt.scatter(X_train_outliers[:,0],X_train_outliers[:,1],label='X_train_outliers')

plt.scatter(X_test_normal[:,0],X_test_normal[:,1],label='X_test_normal')
plt.scatter(X_test_outliers[:,0],X_test_outliers[:,1],label='X_test_outliers')

plt.xlabel('x') #plt.xlabel('Feature 1')
plt.ylabel('y') #plt.ylabel('Feature 2')

plt.legend()
plt.savefig('./NormalAbnormal.png')

plt.show()

#Now we will append the normal points and outliers- train and test separately
X_train=np.append(X_train_normal,X_train_outliers,axis=0)
X_test=np.append(X_test_normal,X_test_outliers,axis=0)

#Training with isolation forest algorithm
clf = IsolationForest(random_state=0, contamination=0.1)
clf.fit(X_train)

#Now we predict the anomaly state for data
y_train=clf.predict(X_train)
y_test=clf.predict(X_test)

# Now we will plot and visualize how good our algorithm works for training data
# y_train(the state) will mark the colors accordingly
plt.figure(figsize=(10, 7.5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

plt.xlabel('x') #plt.xlabel('Feature 1')
plt.ylabel('y') #plt.ylabel('Feature 2')

# This is to set the legend appropriately
legend_elements = [Line2D([], [], marker='o', color='yellow', label='Marked as normal', linestyle='None'),
                   Line2D([], [], marker='o', color='indigo', label='Marked as anomaly', linestyle='None')]
plt.legend(handles=legend_elements)

plt.savefig('./NormalAbnormal2.png')
plt.show()

# Now we will do the same for the test data
plt.figure(figsize=(10, 7.5))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)

#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')

plt.xlabel('x')
plt.ylabel('y')

legend_elements = [Line2D([], [], marker='o', color='yellow', label='Marked as normal', linestyle='None'),
                   Line2D([], [], marker='o', color='indigo', label='Marked as anomaly', linestyle='None')]
plt.legend(handles=legend_elements)

plt.savefig('./NormalAbnormal3.png')
plt.show()



import glob
#import imageio
import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

import PIL
import time
from tensorflow.keras import layers

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

#BUFFER_SIZE = 60000
BUFFER_SIZE = 10000

BATCH_SIZE = 256 # Batch and shuffle the data
#BUFFER_SIZE = 60000 # Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
 model = tf.keras.Sequential()

 model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
 model.add(layers.BatchNormalization())
 model.add(layers.LeakyReLU())

 model.add(layers.Reshape((7, 7, 256)))
 assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

 model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
 assert model.output_shape == (None, 7, 7, 128)
 model.add(layers.BatchNormalization())
 model.add(layers.LeakyReLU())

 model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
 assert model.output_shape == (None, 14, 14, 64)
 model.add(layers.BatchNormalization())
 model.add(layers.LeakyReLU())

 model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
 assert model.output_shape == (None, 28, 28, 1)

 return model

generator = make_generator_model()

def make_discriminator_model():
 model = tf.keras.Sequential()
 model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',

 input_shape=[28, 28, 1]))
 model.add(layers.LeakyReLU())
 model.add(layers.Dropout(0.3))

 model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
 model.add(layers.LeakyReLU())
 model.add(layers.Dropout(0.3))

 model.add(layers.Flatten())
 model.add(layers.Dense(1))

 return model

discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
 real_loss = cross_entropy(tf.ones_like(real_output), real_output)
 fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

 total_loss = real_loss + fake_loss
 return total_loss

def generator_loss(fake_output):
 return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
 discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

#EPOCHS = 50
noise_dim = 100
#num_examples_to_generate = 16

#EPOCHS = 50
EPOCHS = 8

#num_examples_to_generate = 16
num_examples_to_generate = 4

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
 noise = tf.random.normal([BATCH_SIZE, noise_dim])

 with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
 for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
        train_step(image_batch)

        # Produce images for the GIF as we go
        #display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

 # Save the model every 15 epochs
 if (epoch + 1) % 15 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

 print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

 # Generate after the final epoch
 display.clear_output(wait=True)

 # Generate and store images
 generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
 # Notice `training` is set to False.

 # Notice `training` is set to False.
 # This is so all layers run in inference mode (batchnorm).
 predictions = model(test_input, training=False)

 #print(predictions)
 #print(predictions.shape)

 #fig = plt.figure(figsize=(4,4))

 #for i in range(predictions.shape[0]):
    #plt.subplot(4, 4, i+1)
    #plt.imshow(np.array(predictions[i, :, :, 0]) * 127.5 + 127.5, cmap='gray')
    #cv2.imshow('image', predictions[i, :, :, 0] * 127.5 + 127.5)
    #plt.axis('off')

 #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
 #plt.show()

#train(train_dataset, EPOCHS)



# example of loading the fashion_mnist dataset
from keras.datasets.fashion_mnist import load_data

# load the images into memory
(trainX, trainy), (testX, testy) = load_data()

# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

# example of loading the CIFAR-10 dataset
from keras.datasets.cifar10 import load_data

# we load the images into the memory
(trainX, trainy), (testX, testy) = load_data()

# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

#import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot

# plot raw pixel data
pyplot.imshow(trainX[49])
pyplot.show()

# example of loading and plotting the cifar10 dataset
from keras.datasets.cifar10 import load_data
from matplotlib import pyplot

# load the images into memory
(trainX, trainy), (testX, testy) = load_data()

# plot images from the training dataset
for i in range(49):
    # define subplot
    pyplot.subplot(7, 7, 1 + i)

    pyplot.axis('off') # turn off axis
    pyplot.imshow(trainX[i]) # plot raw pixel data

pyplot.show()

# example of defining the discriminator model

# example of defining the discriminator model
from keras.models import Sequential
from keras.optimizers import Adam

from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten

from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()

    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))

    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

model = define_discriminator() # define the model
model.summary() # we now summarize the model

# plot the model
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)

# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
# we use: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

# load cifar10 dataset
(trainX, _), (_, _) = load_data()

# convert from unsigned ints to floats
X = trainX.astype('float32')

# scale from [0,255] to [-1,1]
X = (X - 127.5) / 127.5

# load and prepare cifar10 training images
def load_real_samples():
    (trainX, _), (_, _) = load_data()

    X = trainX.astype("float32") # convert from unsigned ints to floats
    X = (X - 127.5) / 127.5 # scale from [0,255] to [-1,1]

    return X

# select real samples
def generate_real_samples(dataset, n_samples):
    # we choose random instances

	# choose random instances
    ix = randint(0, dataset.shape[0], n_samples)

    X = dataset[ix] # retrieve selected images
    y = ones((n_samples, 1)) # generate 'real' class labels (1)

    return X, y

# generate n fake samples with class labels
def generate_fake_samples(n_samples):
    # generate uniform random numbers in [0,1]
    X = rand(32 * 32 * 3 * n_samples)

    X = -1 + X * 2 # update to have the range [-1, 1]
    X = X.reshape((n_samples, 32, 32, 3)) # reshape into a batch of color images

	# generate 'fake' class labels (0)
    y = zeros((n_samples, 1))

    return X, y

# example of training the discriminator model on real and random cifar10 images
from numpy import expand_dims

from numpy import ones
from numpy import zeros

from numpy.random import rand
from numpy.random import randint
from keras.datasets.cifar10 import load_data

from keras.optimizers import Adam
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten

from keras.layers import Dropout
from keras.layers import LeakyReLU

# train the discriminator model
#def train_discriminator(model, dataset, n_iter=20, n_batch=128):

#def train_discriminator(model, dataset, n_iter=20, n_batch=128):
def train_discriminator(model, dataset, n_iter=8, n_batch=128):
	half_batch = int(n_batch / 2)

	# manually enumerate epochs
	for i in range(n_iter):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)

		# update discriminator on real samples
		_, real_acc = model.train_on_batch(X_real, y_real)

		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(half_batch)

		# update discriminator on fake samples
		_, fake_acc = model.train_on_batch(X_fake, y_fake)

		# summarize the performance
		print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

# define the discriminator model
model = define_discriminator()

# load image data
dataset = load_real_samples()

# fit the model
train_discriminator(model, dataset)

# example of defining the generator model
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Reshape

from keras.layers import Conv2D
from keras.layers import Conv2DTranspose

from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()

    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4

    model.add(Dense(n_nodes, input_dim=latent_dim))

    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))

    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # output layer
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))

    return model

# define the size of the latent space
latent_dim = 100

# define the generator model
model = define_generator(latent_dim)

# summarize the model
model.summary()

# plot the model
plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

from numpy.random import randn # we use randn
# https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
	x_input = randn(latent_dim * n_samples)

    # reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)

	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)

	# predict outputs
	X = g_model.predict(x_input)

	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))

	return X, y

latent_dim = 100 # we define the size of the latent space
model = define_generator(latent_dim) # define the generator model

n_samples = 49 # we now generate samples
X, _ = generate_fake_samples(model, latent_dim, n_samples)

# scale pixel values from [-1,1] to [0,1]
X = (X + 1) / 2.0

# plot the generated samples
for i in range(n_samples):
	# define subplot
	pyplot.subplot(7, 7, 1 + i)

	# turn off axis labels
	pyplot.axis('off')

	# plot single image
	pyplot.imshow(X[i])

# show the figure
pyplot.show()

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False

	# connect them
	model = Sequential()

	# add generator
	model.add(g_model)

	# add the discriminator
	model.add(d_model)

	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model

# size of the latent space
latent_dim = 100

d_model = define_discriminator() # create the discriminator
g_model = define_generator(latent_dim) # create the generator
gan_model = define_gan(g_model, d_model) # create the GAN model

# summarize gan model
gan_model.summary()

# plot gan model
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

# load and prepare cifar10 training images
def load_real_samples():
    # load cifar10 dataset
    (trainX, _), (_, _) = load_data()

    # convert from unsigned ints to floats
    X = trainX.astype('float32')

    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5

    return X

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)

    # retrieve selected images
    X = dataset[ix]

    # generate 'real' class labels (1)
    y = ones((n_samples, 1))

    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)

    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)

    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)

    # predict outputs
    X = g_model.predict(x_input)

    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))

    return X, y

# create and save a plot of generated images
def save_plot(examples, epoch, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0

    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)

        # turn off axis
        pyplot.axis('off')

        # plot raw pixel data
        pyplot.imshow(examples[i])

    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)

    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))

    # save plot
    save_plot(x_fake, epoch)

    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)

# train the generator and the discriminator
#def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):

#def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
#def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=8, n_batch=128):

#def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=3, n_batch=256):
    #bat_per_epo = int(dataset.shape[0] / n_batch)

    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    #print(n_epochs)
    #print(bat_per_epo)

    for i in range(n_epochs): # manually enumerate epochs
        for j in range(bat_per_epo): # enumerate batches over the training set
            X_real, y_real = generate_real_samples(dataset, half_batch) # get randomly selected 'real' samples

            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)

            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)

            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))

            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))

        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)

# size of the latent space
latent_dim = 100

d_model = define_discriminator() # create the discriminator
g_model = define_generator(latent_dim) # create the generator
gan_model = define_gan(g_model, d_model) # create the GAN model

# load image data
dataset = load_real_samples()

# train the model
#train(g_model, d_model, gan_model, dataset, latent_dim)

#train(g_model, d_model, gan_model, dataset, latent_dim)
#train(g_model, d_model, gan_model, dataset, latent_dim)

# example of loading the generator model and generating images
from keras.models import load_model

from matplotlib import pyplot
from numpy.random import randn

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)

    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# create and save a plot of generated images
def save_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)

        # turn off axis
        pyplot.axis('off')

        # plot raw pixel data
        pyplot.imshow(examples[i, :, :])

    pyplot.show()

# load model
#model = load_model('generator_model_200.h5')

#model = load_model('generator_model_200.h5')
#model = load_model('generator_model_200.h5')

# generate images
latent_points = generate_latent_points(100, 100)

# generate images
X = model.predict(latent_points)

X = (X + 1) / 2.0 # scale from [-1,1] to [0,1]
save_plot(X, 10) # plot the result

# an example of generating an image for a specific point in the latent space
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
# use: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

from numpy import asarray
from matplotlib import pyplot
from keras.models import load_model

# load model
#model = load_model('generator_model_200.h5')

#model = load_model('generator_model_200.h5')
#model = load_model('generator_model_200.h5')

# all 0s
vector = asarray([[0.75 for _ in range(100)]])

# generate image
X = model.predict(vector)

# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0

# plot the result
pyplot.imshow(X[0, :, :])
pyplot.show()



# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = './data'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
# use: https://cs.stanford.edu/~acoates/stl10/

# path to the binary train file with image data
DATA_PATH = './data/stl10_binary/train_X.bin'

# path to the binary train file with labels
LABEL_PATH = './data/stl10_binary/train_y.bin'

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """

    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)

        return labels

def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks

        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the images are stored
        # in "column-major order", meaning that "the first 96*96 values are
        # the red channel, the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends on the input file,
        # and this way numpy determines the size on its own.

        # We force the data into 3x96x96 chunks.
        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.

        # Transpose the images
        images = np.transpose(images, (0, 3, 2, 1))

        return images

def read_single_image(image_file):
    """
    This method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """

    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)

    # force into image matrix
    image = np.reshape(image, (3, 96, 96))

    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.

    # transpose to standard format
    image = np.transpose(image, (2, 1, 0))

    return image

def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    """

    plt.imshow(image)
    plt.show()

def save_image(image, name):
    imsave("%s.png" % name, image, format="png")

def download_and_extract():
    """
    Download and extract the STL-10 dataset
    """

    dest_directory = DATA_DIR

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)

        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def save_images(images, labels):
    #print("Saving images to disk")

    #print("Saving images to disk")
    print("Save images to disk")

    i = 0
    for image in images:
        label = labels[i]
        directory = './img/' + str(label) + '/'

        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass

        filename = directory + str(i)

        print(filename)
        save_image(image, filename)

        i = i + 1

if __name__ == "__main__":
    # download the data

    # if needed, download the data
    download_and_extract()

    # test to check if the image is read correctly
    with open(DATA_PATH) as f:
        image = read_single_image(f)
        plot_image(image)

    # test to check if the whole dataset is read correctly
    images = read_all_images(DATA_PATH)
    print(images.shape)

    labels = read_labels(LABEL_PATH)
    print(labels.shape)

    # save images to disk
    #save_images(images, labels)

import os
import tempfile

import subprocess
import numpy as np
import matplotlib.pyplot as plt

#g1 = [[0.9040965370370371, 0.74461, 0.5155018796296296, 0.6067401574074074, 0.5356445370370371, 0.6033470277777778, 0.8600094629629629, 0.5770984814814815, 0.7734221574074074, 0.6068066759259259], [0.9612778981481482, 0.6687222777777778, 0.5260006481481482, 0.5885034444444445, 0.6129901111111111, 0.6673065462962963, 0.7511156203703704, 0.5420412592592593, 0.85680575, 0.6509693703703704], [0.9332610555555555, 0.7233965462962962, 0.48850834259259257, 0.5622162962962962, 0.6895320833333334, 0.6664632407407407, 0.5627155, 0.5813406481481481, 0.8163357314814814, 0.6714644166666667]]
#g2 = [[0.9621442407407408, 0.74461, 0.5344123518518518, 0.6354489166666666, 0.7596060092592593, 0.6634529444444445, 0.8600094629629629, 0.6152494722222223, 0.8769023425925926, 0.6864652407407408], [0.9612778981481482, 0.704607611111111, 0.5458616759259258, 0.6221637685185185, 0.8161331388888889, 0.6673065462962963, 0.7942182962962963, 0.6073334722222222, 0.8597013055555556, 0.6749123333333333], [0.962578675925926, 0.7233965462962962, 0.5377918888888888, 0.647348361111111, 0.7773921296296297, 0.69790375, 0.8423532499999999, 0.5882901481481482, 0.872712101851852, 0.6799068518518518]]
#g3 = [[3.69779109954834, 3.76554012298584, 3.775956630706787, 3.870365619659424, 3.716261386871338, 3.994929790496826, 3.925158977508545, 4.060425758361816, 3.703930377960205, 3.95566463470459], [3.7598299980163574, 3.8211655616760254, 3.7877893447875977, 3.7814974784851074, 3.9992785453796387, 3.978555202484131, 3.631882667541504, 4.439399242401123, 3.957340717315674, 3.800053596496582], [3.9581727981567383, 3.675556182861328, 4.179351329803467, 3.965294361114502, 3.7168478965759277, 3.8895392417907715, 4.040830135345459, 6.364235877990723, 4.151716232299805, 3.7189888954162598]]

#g1 = [[0.7889737216613737, 0.4376289340238787, 0.4409219517892701, 0.4595648499901688, 0.5545440423978997, 0.5206263764533741, 0.7586791255690835, 0.48162266355712213, 0.6732045322341802, 0.4389286757081857], [0.7660555562078939, 0.48029352444285156, 0.4568258605673837, 0.45700917501541244, 0.5736049690514815, 0.5269126901596681, 0.5820140071785596, 0.43289228383491585, 0.703710630924051, 0.449899486301065]]
#g2 = [[0.93355385566179, 0.5170235999151753, 0.4662169285927492, 0.5251649760532959, 0.6554190394805207, 0.6006760280963213, 0.8124006550390338, 0.48162266355712213, 0.7996720828527245, 0.5187196519454024], [0.775196439320086, 0.5117902979187909, 0.4857918543069639, 0.4945914277633235, 0.6793749127030748, 0.5535922006444878, 0.674084747530882, 0.4829471835856364, 0.781411717399264, 0.5201429516294148]]
#g3 = [[5.302708148956299, 6.873373985290527, 8.551833629608154, 4.790539741516113, 9.20945405960083, 7.877237796783447, 8.614120483398438, 8.584418296813965, 8.528287410736084, 8.6995530128479], [9.554312229156494, 4.544234275817871, 4.635648727416992, 4.945080280303955, 4.576082229614258, 5.378477573394775, 5.201163291931152, 4.78830099105835, 4.979913234710693, 4.77419376373291]]

#g1 = [[0.0321416857986225, 0.20104712041884817, 0.019455894476504535, 0.011900826446280993, 0.022086698533047632, 0.04468016714882674, 0.011269472986410341, 0.037815126050420166, 0.10298273155416014, 0.0], [0.04131424853610931, 0.1913848744212151, 0.00895819508958195, 0.017110891740704178, 0.022086698533047632, 0.09772798008092128, 0.007637390004980906, 0.07848484848484849, 0.1090909090909091, 0.0003318951211417192], [0.029223444426202592, 0.06714701824313966, 0.03366563163915673, 0.014151719598486094, 0.02045529528208512, 0.04526349822179115, 0.008959681433549029, 0.07848484848484849, 0.1090909090909091, 0.0003318951211417192]]
#g2 = [[0.13096904650801058, 0.35081374321880654, 0.06159246848571885, 0.30612722170252576, 0.03568505483712556, 0.33062812673707614, 0.04782820888238165, 0.27070457354758964, 0.35906172192373215, 0.23543457497612225], [0.12003722084367247, 0.30970504281636535, 0.055332153771915714, 0.13032440056417488, 0.033753891528756345, 0.2395806699053951, 0.0522642428177244, 0.3875333196637277, 0.2628873141586772, 0.18817852834740653], [0.28604802076573654, 0.3537072043688301, 0.11930355791067372, 0.2977812816758644, 0.03504175536269854, 0.4955348660459813, 0.05542051531356344, 0.3875333196637277, 0.2628873141586772, 0.18817852834740653]]
#g3 = [[7.766354084014893, 6.104092597961426, 8.336114883422852, 6.420071125030518, 6.815962791442871, 5.024633407592773, 6.99282169342041, 4.712412357330322, 7.170934677124023, 7.1436285972595215], [7.420742511749268, 5.551357269287109, 5.628402233123779, 6.542465686798096, 6.72459602355957, 6.300387382507324, 7.186479568481445, 6.405231952667236, 6.526792049407959, 4.3767595291137695], [5.363500118255615, 6.78480863571167, 6.5532684326171875, 7.049682140350342, 4.75616455078125, 6.0572052001953125, 6.733193397521973, 6.405231952667236, 6.526792049407959, 4.3767595291137695]]

#g1 = [[0.8322690647579725, 0.46508818965244675, 0.4554794384184438, 0.46258671669986035, 0.5596197878676594, 0.5196174763431345, 0.4960934597989699, 0.4145508814661052, 0.6760784715597402, 0.4410497488078411], [0.8079101918260816, 0.46472301049649256, 0.46522895678705267, 0.4806114442137722, 0.5461513184154637, 0.5166404570479255, 0.5893097072368136, 0.4432133749487719, 0.7390971882169581, 0.45779990708967033], [0.8869100692278536, 0.47937692396718956, 0.4679810410710097, 0.43955356472805, 0.5039657981286474, 0.5058497120286687, 0.555222209456331, 0.4432133749487719, 0.7390971882169581, 0.45779990708967033]]
#g2 = [[0.8994543282452314, 0.50110293822674, 0.47136620220671804, 0.5072057519442077, 0.729771268038981, 0.5376782378981044, 0.7162969839053682, 0.46568174567064347, 0.795980885425057, 0.502347495496864], [0.9247059927964847, 0.5554989020025276, 0.4742671741781292, 0.5158302646403715, 0.6092528939240605, 0.5507575431405519, 0.795656957149063, 0.48186382479469303, 0.7900468243678925, 0.514806566514857], [0.9454696185574382, 0.48467452815862666, 0.5146056300735227, 0.5381002639846892, 0.6532587673252515, 0.5587824122581884, 0.7986278014575245, 0.48186382479469303, 0.7900468243678925, 0.514806566514857]]
#g3 = [[7.816967964172363, 7.423698902130127, 4.944567680358887, 5.794970989227295, 6.627023220062256, 6.498100757598877, 6.7436909675598145, 6.511788368225098, 5.947117805480957, 7.172915935516357], [6.732392311096191, 6.878643035888672, 6.776483058929443, 4.586198329925537, 8.029012680053711, 6.286039352416992, 5.112016201019287, 7.594101428985596, 8.815820217132568, 6.8603515625], [6.7368292808532715, 7.183847427368164, 6.407506465911865, 8.1144118309021, 5.538482666015625, 7.131941318511963, 8.118283748626709, 7.594101428985596, 8.815820217132568, 6.8603515625]]

#g1 = np.mean(g1, axis=0)
#g2 = np.mean(g2, axis=0)
#g3 = np.mean(g3, axis=0)

# This is also for manualseed = None:
#g1 = [0.9271919444444445, 0.5885791851851851, 0.5393023240740741, 0.5128948611111112, 0.669631712962963, 0.6479999444444444, 0.6850078611111112, 0.5864626481481482, 0.8149177685185185, 0.643201712962963]
#g2 = [0.9590519166666667, 0.7176374351851852, 0.5483709814814816, 0.6214027777777777, 0.7848986296296296, 0.6873968611111112, 0.9012428240740741, 0.5987799074074074, 0.8671639166666667, 0.6892931388888888]
#g3 = [5.915262699127197, 4.76083517074585, 4.176671504974365, 6.598055362701416, 6.626327037811279, 4.304831027984619, 4.264116287231445, 6.072173118591309, 5.856869220733643, 4.278533458709717]

# This is for manualseed = None:
#g1 = [0.09730586370839936, 0.14985835694050992, 0.020115416323165703, 0.018390804597701153, 0.025341451374033243, 0.029450261780104712, 0.013573911604039068, 0.09443354955498567, 0.17894581426251294, 0.0019766101136550816]
#g2 = [0.15686274509803924, 0.36533775221756504, 0.11579892280071813, 0.22565922920892498, 0.03310932633994427, 0.43326133909287257, 0.04973183812774256, 0.28463530427276656, 0.230287859824781, 0.24048525214081828]
#g3 = [6.9103169441223145, 4.698286056518555, 6.453959941864014, 6.823458671569824, 4.304108619689941, 6.651432514190674, 4.146397113800049, 6.246163845062256, 6.551299095153809, 4.729597568511963]

# manualseed = 1:
#g1 = [0.9431644537037038, 0.5947754722222222, 0.5026536759259259, 0.5529489351851852, 0.5820512407407408, 0.6237469722222222, 0.6526065, 0.5458433333333333, 0.7619122222222222, 0.6784945462962964]
#g2 = [0.9593461574074074, 0.6894443240740741, 0.5706896666666668, 0.6520250833333334, 0.7297374444444444, 0.668991675925926, 0.864206925925926, 0.5770702407407406, 0.838014925925926, 0.7042570555555556]
#g3 = [7.1082305908203125, 6.64898157119751, 4.659130573272705, 7.035958766937256, 4.7405314445495605, 6.410126686096191, 7.339005470275879, 6.416282653808594, 4.540286064147949, 4.528176784515381]

# manualseed = 1:
g1 = [0.033753891528756345, 0.03884430176565008, 0.02499177902005919, 0.013210039630118891, 0.026640355204736062, 0.06336131791541265, 0.008959681433549029, 0.06388587377888044, 0.09978240596829345, 0.031672100605496044]
g2 = [0.14316635745207174, 0.2295200288704439, 0.20970625798212003, 0.23088531187122735, 0.02955665024630542, 0.36083786598773293, 0.04461814036801824, 0.27184238301864666, 0.28747894441325095, 0.1475853187379266]
g3 = [6.700444221496582, 6.687617301940918, 7.262628078460693, 6.6029953956604, 4.224178791046143, 6.782968044281006, 7.263069152832031, 6.867167949676514, 4.703798294067383, 7.097740173339844]

#print(g1.shape)
arrayLoop = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

res21_total = g1
best_auc21_total = g2
res22_total = g3

plt.figure(1)
#plt.plot(np.array(arrayLoop), res21_total, 'b-o', arrayLoop, best_auc21_total, 'r-o')
#plt.xticks(range(len(arrayLoop)), arrayLoop)

#print(res21_total)
#res21_total = list(res21_total)
#best_auc21_total = list(best_auc21_total)
#res22_total = list(res22_total)

#res21_total.reverse()
#best_auc21_total.reverse()
#res22_total.reverse()

res21_total = np.array(res21_total)
best_auc21_total = np.array(best_auc21_total)
res22_total = np.array(res22_total)

plt.plot(range(len(arrayLoop)), res21_total, 'bo', range(len(arrayLoop)), best_auc21_total, 'rx')
plt.xticks(range(len(arrayLoop)), arrayLoop)
plt.plot(res21_total, 'b-o', best_auc21_total, 'r-x')

#plt.ylabel('AUC')
plt.xlabel('Anomaly Class')

#plt.ylabel('AUC')
plt.ylabel('F1 Score')

#plt.ylabel('F1 Score')
#plt.ylabel('AUPRC')

#plt.legend(['AUC', 'Best AUC'])
plt.legend(['F1 Score', 'Best F1 Score'])

#plt.legend(['F1 Score', 'Best F1 Score'])
#plt.legend(['AUPRC', 'Best AUPRC'])
plt.show()

plt.figure(2)
#plt.plot(np.array(arrayLoop), res22_total, 'b-o')
#plt.xticks(range(len(arrayLoop)), arrayLoop)

plt.plot(range(len(arrayLoop)), res22_total, 'bo')
plt.xticks(range(len(arrayLoop)), arrayLoop)
plt.plot(res22_total, 'b-o')

plt.ylabel('Avg Run Time (ms/batch)')
plt.xlabel('Anomaly Class')
plt.show()



import tarfile
import numpy as np
import sonnet as snt
import tensorflow as tf

from six.moves import xrange
from six.moves import urllib
from six.moves import cPickle

# https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
data_path = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" # CIFAR-10 Dataset
# we use: https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb

local_data_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(local_data_dir)

url = urllib.request.urlopen(data_path)
archive = tarfile.open(fileobj=url, mode='r|gz')

archive.extractall(local_data_dir)
url.close()

archive.close()
print('extracted data files to %s' % local_data_dir)

def unpickle(filename):
    with open(filename, 'rb') as fo:
        return cPickle.load(fo, encoding='latin1')

def reshape_flattened_image_batch(flat_image_batch):
    return flat_image_batch.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])  # convert from NCHW to NHWC

def combine_batches(batch_list):
    images = np.vstack([reshape_flattened_image_batch(batch['data'])
                        for batch in batch_list])

    labels = np.vstack([np.array(batch['labels']) for batch in batch_list]).reshape(-1, 1)
    return {'images': images, 'labels': labels}

train_data_dict = combine_batches([
    unpickle(os.path.join(local_data_dir,
                          'cifar-10-batches-py/data_batch_%d' % i))
    for i in range(1, 5)])

valid_data_dict = combine_batches([unpickle(os.path.join(local_data_dir, 'cifar-10-batches-py/data_batch_5'))])
test_data_dict = combine_batches([unpickle(os.path.join(local_data_dir, 'cifar-10-batches-py/test_batch'))])

def cast_and_normalise_images(data_dict):
  """Convert images to floating point with the range [0.5, 0.5]"""
  images = data_dict['images']

  data_dict['images'] = (tf.cast(images, tf.float32) / 255.0) - 0.5
  return data_dict

data_variance = np.var(train_data_dict['images'] / 255.0)

def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
    for i in range(num_residual_layers):
        h_i = tf.nn.relu(h)

        h_i = snt.Conv2D(
            output_channels=num_residual_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="res3x3_%d" % i)(h_i)
        h_i = tf.nn.relu(h_i)

        h_i = snt.Conv2D(
            output_channels=num_hiddens,
            kernel_shape=(1, 1),
            stride=(1, 1),
            name="res1x1_%d" % i)(h_i)
        h += h_i

    return tf.nn.relu(h)

class Encoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name='encoder'):
        super(Encoder, self).__init__(name=name)

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens / 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")(x)
        h = tf.nn.relu(h)

        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")(h)
        h = tf.nn.relu(h)

        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_3")(h)

        h = residual_stack(h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

        return h

class Decoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name='decoder'):
        super(Decoder, self).__init__(name=name)

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1")(x)

        h = residual_stack(h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

        h = snt.Conv2DTranspose(
            output_channels=int(self._num_hiddens / 2),
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")(h)
        h = tf.nn.relu(h)

        x_recon = snt.Conv2DTranspose(
            output_channels=3,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3")(h)

        return x_recon

tf.reset_default_graph()
# VQ-VAE, DeepMind, Google AI
# https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb

# Set hyper-parameters
batch_size = 32
image_size = 32

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU
#num_training_updates = 50000

#num_training_updates = 50000
num_training_updates = 100

num_hiddens = 128
num_residual_layers = 2
num_residual_hiddens = 32

# These hyper-parameters define the size of the model (number of parameters and layers).
# The hyper-parameters in the paper were (For ImageNet):
# batch_size = 128
# image_size = 128
# num_hiddens = 128
# num_residual_hiddens = 32
# num_residual_layers = 2

# This value is usually 64. This value will not change the capacity in the information-bottleneck.
embedding_dim = 64

# The higher this value, the higher the capacity in the information bottleneck.
num_embeddings = 512

# commitment_cost should be set appropriately. It's often useful to try a couple of values. It mostly depends on
# the scale of the reconstruction cost (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.
commitment_cost = 0.25

# Use EMA updates for the codebook (instead of the Adam optimizer). This typically converges faster, and makes
# the model less dependent on choice of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
vq_use_ema = False

decay = 0.99
learning_rate = 3e-4

# Data Loading
train_dataset_iterator = (
    tf.data.Dataset.from_tensor_slices(train_data_dict)
        .map(cast_and_normalise_images)
        .shuffle(10000)
        .repeat(-1)  # repeat indefinitely
        .batch(batch_size)).make_one_shot_iterator()

valid_dataset_iterator = (
    tf.data.Dataset.from_tensor_slices(valid_data_dict)
        .map(cast_and_normalise_images)
        .repeat(1)  # 1 epoch
        .batch(batch_size)).make_initializable_iterator()

train_dataset_batch = train_dataset_iterator.get_next()
valid_dataset_batch = valid_dataset_iterator.get_next()

def get_images(sess, subset='train'):
    if subset == 'train':
        return sess.run(train_dataset_batch)['images']
    elif subset == 'valid':
        return sess.run(valid_dataset_batch)['images']

# we build the modules
encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)

pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
                          kernel_shape=(1, 1), stride=(1, 1), name="to_vq")

if vq_use_ema:
    vq_vae = snt.nets.VectorQuantizerEMA(embedding_dim=embedding_dim,
        num_embeddings=num_embeddings, commitment_cost=commitment_cost, decay=decay)
else:
    vq_vae = snt.nets.VectorQuantizer(embedding_dim=embedding_dim,
        num_embeddings=num_embeddings, commitment_cost=commitment_cost)

# Process inputs with conv stack, finishing with 1x1 to get to correct size.
x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
z = pre_vq_conv1(encoder(x))

# vq_output_train["quantize"] are the quantized outputs of the encoder.
# That is also what is used during training with the straight-through estimator.
# To get the one-hot coded assignments use vq_output_train["encodings"] instead.

# These encodings will not pass gradients into to encoder,
# but can be used to train a PixelCNN on top afterwards.

vq_output_train = vq_vae(z, is_training=True) # For training
x_recon = decoder(vq_output_train["quantize"]) # Training

recon_error = tf.reduce_mean((x_recon - x) ** 2) / data_variance  # Normalized MSE
loss = recon_error + vq_output_train["loss"]

# For evaluation, make sure is_training=False!
vq_output_eval = vq_vae(z, is_training=False)
x_recon_eval = decoder(vq_output_eval["quantize"])

# The following is a useful value to track during training.
# It indicates how many codes are 'active' on average.
perplexity = vq_output_train["perplexity"]

# Create optimizer and TF session.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss)
sess = tf.train.SingularMonitoredSession()

# Train
train_res_recon_error = []
train_res_perplexity = []

for i in xrange(num_training_updates):
    feed_dict = {x: get_images(sess)}

    results = sess.run([train_op, recon_error, perplexity],
                       feed_dict=feed_dict)

    train_res_recon_error.append(results[1])
    train_res_perplexity.append(results[2])

    if (i + 1) % 100 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))

        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()

f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_res_recon_error)

ax.set_yscale('log')
ax.set_title('NMSE.')

ax = f.add_subplot(1,2,2)
ax.plot(train_res_perplexity)
ax.set_title('Average codebook usage (perplexity).')

# Reconstructions
sess.run(valid_dataset_iterator.initializer)

train_originals = get_images(sess, subset='train')
train_reconstructions = sess.run(x_recon_eval, feed_dict={x: train_originals})

valid_originals = get_images(sess, subset='valid')
valid_reconstructions = sess.run(x_recon_eval, feed_dict={x: valid_originals})

def convert_batch_to_image_grid(image_batch):
  reshaped = (image_batch.reshape(4, 8, 32, 32, 3)
              .transpose(0, 2, 1, 3, 4)
              .reshape(4 * 32, 8 * 32, 3))
  return reshaped + 0.5

f = plt.figure(figsize=(16,8))
ax = f.add_subplot(2,2,1)

ax.imshow(convert_batch_to_image_grid(train_originals), interpolation='nearest')
ax.set_title('training data originals')
plt.axis('off')

ax = f.add_subplot(2,2,2)
ax.imshow(convert_batch_to_image_grid(train_reconstructions), interpolation='nearest')

ax.set_title('training data reconstructions')
plt.axis('off')

ax = f.add_subplot(2,2,3)
ax.imshow(convert_batch_to_image_grid(valid_originals), interpolation='nearest')

ax.set_title('validation data originals')
plt.axis('off')

ax = f.add_subplot(2,2,4)
ax.imshow(convert_batch_to_image_grid(valid_reconstructions), interpolation='nearest')

ax.set_title('validation data reconstructions')
plt.axis('off')
plt.pause(2)



import torch
import torch.nn as nn

input_dim = 5
hidden_dim = 10

n_layers = 1
lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# we use: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/

batch_size = 1
seq_len = 1

inp = torch.randn(batch_size, seq_len, input_dim)
hidden_state = torch.randn(n_layers, batch_size, hidden_dim)

cell_state = torch.randn(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)

out, hidden = lstm_layer(inp, hidden)
print("Output shape: ", out.shape)
print("Hidden: ", hidden)

seq_len = 3
inp = torch.randn(batch_size, seq_len, input_dim)

out, hidden = lstm_layer(inp, hidden)
print(out.shape)

# Obtaining the last output
out = out.squeeze()[-1, :]
print(out.shape)

# we now use: https://github.com/gabrielloye/LSTM_Sentiment-Analysis
# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/

#import re
#import bz2
#import nltk

#import numpy as np
#nltk.download('punkt')
#from collections import Counter

#train_file = bz2.BZ2File('../input/amazon_reviews/train.ft.txt.bz2')
#test_file = bz2.BZ2File('../input/amazon_reviews/test.ft.txt.bz2')
#train_file = train_file.readlines()
#test_file = test_file.readlines()

import bz2
from collections import Counter

import re
import nltk

import numpy as np
nltk.download('punkt')

train_file = bz2.BZ2File('../input/amazon_reviews/train.ft.txt.bz2')
test_file = bz2.BZ2File('../input/amazon_reviews/test.ft.txt.bz2')

train_file = train_file.readlines()
test_file = test_file.readlines()

num_train = 800000  # We're training on the first 800,000 reviews in the dataset
num_test = 200000  # Using 200,000 reviews from test set

train_file = [x.decode('utf-8') for x in train_file[:num_train]]
test_file = [x.decode('utf-8') for x in test_file[:num_test]]

# Extracting labels from sentences
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]

test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

# Some simple cleaning of data
for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d', '0', train_sentences[i])

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d', '0', test_sentences[i])

# Modify URLs to <url>
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in \
            train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in \
            test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

words = Counter()  # Dictionary that will map a word to the number of times it appeared in all the training sentences

for i, sentence in enumerate(train_sentences):
    train_sentences[i] = []

    for word in nltk.word_tokenize(sentence):  # Tokenize the words
        words.update([word.lower()])  # Convert all the words to lowercase
        train_sentences[i].append(word)

    if i%20000 == 0:
        print(str((i*100)/num_train) + "% done")

print("OK. Done.")

# Remove the words that only appear once
words = {k:v for k,v in words.items() if v>1}

# Sorting the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)

# Adding padding and unknown to our vocabulary so that they will be assigned an index
words = ['_PAD','_UNK'] + words

# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

for i, sentence in enumerate(train_sentences):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

for i, sentence in enumerate(test_sentences):
    # For test sentences, we have to tokenize the sentences as well
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)

    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]

    return features

seq_len = 200  # The length that the sentences will be padded/shortened to
train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)

# Converting our labels into numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

split_frac = 0.5 # 50% validation, 50% test
split_id = int(split_frac * len(test_sentences))

val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

import torch
import torch.nn as nn # use torch and nn
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

batch_size = 400 # we now define the batch size
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
#is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
#if is_cuda:
#    device = torch.device("cuda")
#else:
#    device = torch.device("cpu")

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        x = x.long()
        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden

vocab_size = len(word2idx) + 1
output_size = 1

embedding_dim = 400
hidden_dim = 512
n_layers = 2

model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.to(device)

lr=0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

clip = 5
epochs = 2
counter = 0
print_every = 1000
valid_loss_min = np.Inf

model.train()
for i in range(epochs):
    h = model.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1

        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)

        model.zero_grad()
        output, h = model(inputs, h)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []

            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])

                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)

                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(i + 1, epochs), "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()), "Val Loss: {:.6f}".format(np.mean(val_losses)))

            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

# Loading the best model
model.load_state_dict(torch.load('./state_dict.pt'))

num_correct = 0
test_losses = []
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])

    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)

    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    pred = torch.round(output.squeeze())  # Rounds the output to 0/1

    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())

    num_correct += np.sum(correct)

# use: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
print("Test loss: {:.3f}".format(np.mean(test_losses)))

test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))



import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data

# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959
# use: https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959

# MNIST: Four files are available on this site:
# train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
# train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
# t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
# t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

# From Terminal: (1) cd mnist/
# (2) gzip train-images-idx3-ubyte.gz -d
# (3) gzip train-labels-idx1-ubyte.gz -d
# (4) gzip t10k-images-idx3-ubyte.gz -d
# (5) gzip t10k-labels-idx1-ubyte.gz -d

import os
import struct

def load_mnist(path2, kind='train'):
    labels_path = os.path.join(path2)
    images_path = os.path.join(path2)

from mlxtend.data import loadlocal_mnist # we load the data
#X_train, y_train = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/', kind='train')

#y_train = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/train-labels-idx1-ubyte')
#X_train = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/train-images-idx3-ubyte')
#print('Rows: {X_train.shape[0]},  Columns: {X_train.shape[1]}')

X_train, y_train = loadlocal_mnist(images_path='/Users/dionelisnikolaos/Downloads/mnist/train-images-idx3-ubyte',
        labels_path='/Users/dionelisnikolaos/Downloads/mnist/train-labels-idx1-ubyte')

#X_test, y_test = load_mnist('./mnist/', kind='t10k') # loading the data
#y_test = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/t10k-labels-idx1-ubyte')
#X_test = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/t10k-images-idx3-ubyte')
#print('Rows: {X_test.shape[0]},  Columns: {X_test.shape[1]}')

X_test, y_test = loadlocal_mnist(images_path='/Users/dionelisnikolaos/Downloads/mnist/t10k-images-idx3-ubyte',
        labels_path='/Users/dionelisnikolaos/Downloads/mnist/t10k-labels-idx1-ubyte')

mean_vals = np.mean(X_train, axis=0) # mean centering
std_val = np.std(X_train) # normalization

#print(X_train.shape)
#print(y_train.shape)

#print(X_test.shape)
#print(y_test.shape)

#print(mean_vals.shape)
#print(std_val.shape)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10):
    img = X_train_centered[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_yticks([])
ax[0].set_xticks([])

plt.tight_layout()
plt.show()

np.random.seed(123)
tf.set_random_seed(123)

import tensorflow.contrib.keras as keras
y_train_onehot = keras.utils.to_categorical(y_train)

print('First 3 labels: ', y_train[:3])
print('\nFirst 3 labels (one-hot):\n', y_train_onehot[:3])

# https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959
y_train_onehot = keras.utils.to_categorical(y_train)

print('First 3 labels: ', y_train[:3])
print('\nFirst 3 labels (one-hot):\n', y_train_onehot[:3])

# initialize the model
model = keras.models.Sequential()

# add the input layer
model.add(keras.layers.Dense(units=50, input_dim=X_train_centered.shape[1],
    kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh'))

model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

history = model.fit(X_train_centered, y_train_onehot, batch_size=64, epochs=50, verbose=1, validation_split=0.1)
y_train_pred = model.predict_classes(X_train_centered, verbose=0) # we train the model
print('First 3 predictions: ', y_train_pred[:3])

# calculate training accuracy
y_train_pred = model.predict_classes(X_train_centered, verbose=0)

correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]

#print('Training accuracy: {(train_acc * 100):.2f}')
print(train_acc)

# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# use: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]

# we calculate the testing accuracy
# First 3 predictions: [5 0 4], 0.9799666666666667, 0.9621
print(test_acc) # First 3 predictions: [5 0 4], 0.9799666666666667, 0.9621

# 48192/54000 [=========================>....] - ETA: 0s - loss: 0.0803 - acc: 0.9801
# 49984/54000 [==========================>...] - ETA: 0s - loss: 0.0800 - acc: 0.9802
# 51904/54000 [===========================>..] - ETA: 0s - loss: 0.0798 - acc: 0.9802
# 53952/54000 [============================>.] - ETA: 0s - loss: 0.0794 - acc: 0.9802
# 54000/54000 [==============================] - 2s 28us/sample - loss: 0.0794 - acc: 0.9803 - val_loss: 0.1108 - val_acc: 0.9668



ds = tf.contrib.distributions

def sample_mog(batch_size, n_mixture=8, std=0.01, radius=1.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture)

    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))

    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)

    return data.sample(batch_size)

#sample_mog(128)
#print(sample_mog(128))

samplePoints = sample_mog(1000)
#print(samplePoints)

#plt.plot([1,2,3,4])
import matplotlib.pyplot as plt
#plt.plot(samplePoints[:,0], samplePoints[:,1])

tf.InteractiveSession()
samplePoints2 = samplePoints.eval()
plt.plot(samplePoints2[:,0], samplePoints2[:,1])

plt.xlabel('x')
plt.ylabel('y')
plt.show()

# np.exp(a)/np.sum(np.exp(a))
# use: np.exp(a)/np.sum(np.exp(a))

# https://github.com/samet-akcay/ganomaly
# GANomaly: https://github.com/samet-akcay/ganomaly

# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.875 AUC: 0.533 max AUC: 0.559
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.830 AUC: 0.531 max AUC: 0.559
# >> Training model Ganomaly.[Done]

# Namespace(anomaly_class='bird', batchsize=64, beta1=0.5, dataroot='', dataset='cifar10',
# device='gpu', display=False, display_id=0, display_port=8097, display_server='http://localhost',
# droplast=True, extralayers=0, gpu_ids=[0], isTrain=True, isize=32, iter=0, load_weights=False, lr=0.0002,
# manualseed=-1, metric='roc', model='ganomaly', name='ganomaly/cifar10', nc=3, ndf=64, ngf=64, ngpu=1, niter=15,
# nz=100, outf='./output', phase='train', print_freq=100, proportion=0.1, resume='', save_image_freq=100,
# save_test_images=False, w_bce=1, w_enc=1, w_rec=50, workers=8)

# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 4.057 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 4.791 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 3/15
#    Avg Run Time (ms/batch): 4.897 AUC: 0.519 max AUC: 0.519
# >> Training model Ganomaly. Epoch 4/15
#    Avg Run Time (ms/batch): 4.792 AUC: 0.502 max AUC: 0.519
# >> Training model Ganomaly. Epoch 5/15
#    Avg Run Time (ms/batch): 4.937 AUC: 0.536 max AUC: 0.536
# >> Training model Ganomaly. Epoch 6/15
#    Avg Run Time (ms/batch): 4.883 AUC: 0.498 max AUC: 0.536
# >> Training model Ganomaly. Epoch 7/15
#    Avg Run Time (ms/batch): 4.960 AUC: 0.503 max AUC: 0.536
# >> Training model Ganomaly. Epoch 8/15
#    Avg Run Time (ms/batch): 4.916 AUC: 0.559 max AUC: 0.559
# >> Training model Ganomaly. Epoch 9/15
#    Avg Run Time (ms/batch): 4.870 AUC: 0.522 max AUC: 0.559
# >> Training model Ganomaly. Epoch 10/15
#    Avg Run Time (ms/batch): 4.898 AUC: 0.539 max AUC: 0.559
#  65% 455/703 [00:16<00:08, 28.19it/s]
# >> Training model Ganomaly. Epoch 11/15
#    Avg Run Time (ms/batch): 4.900 AUC: 0.529 max AUC: 0.559
# >> Training model Ganomaly. Epoch 12/15
#    Avg Run Time (ms/batch): 4.856 AUC: 0.541 max AUC: 0.559
# >> Training model Ganomaly. Epoch 13/15
#    Avg Run Time (ms/batch): 4.910 AUC: 0.528 max AUC: 0.559
# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.875 AUC: 0.533 max AUC: 0.559
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.830 AUC: 0.531 max AUC: 0.559
# >> Training model Ganomaly.[Done]



# https://github.com/samet-akcay/ganomaly
# we use: https://github.com/samet-akcay/ganomaly
#    Avg Run Time (ms/batch): 274.149 AUC: 0.621 max AUC: 0.621
#    >> Avg Run Time (ms/batch): 284.825 AUC: 0.649 max AUC: 0.649

# Namespace(anomaly_class='bird', batchsize=64, beta1=0.5, dataroot='', dataset='cifar10', device='gpu', display=False,
# display_id=0, display_port=8097, display_server='http://localhost', droplast=True, extralayers=0, gpu_ids=[0],
# isTrain=True, isize=32, iter=0, load_weights=False, lr=0.0002, manualseed=-1, metric='roc', model='ganomaly',
# name='ganomaly/cifar10', nc=3, ndf=64, ngf=64, ngpu=1, niter=15, nz=100, outf='./output', phase='train', print_freq=100,
# proportion=0.1, resume='', save_image_freq=100, save_test_images=False, w_bce=1, w_enc=1, w_rec=50, workers=8)

# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 4.100 AUC: 0.504 max AUC: 0.504
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 4.894 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 3/15
#    Avg Run Time (ms/batch): 4.904 AUC: 0.491 max AUC: 0.513
# >> Training model Ganomaly. Epoch 4/15
#    Avg Run Time (ms/batch): 4.850 AUC: 0.538 max AUC: 0.538
# >> Training model Ganomaly. Epoch 5/15
#    Avg Run Time (ms/batch): 4.849 AUC: 0.498 max AUC: 0.538
# >> Training model Ganomaly. Epoch 6/15
#    Avg Run Time (ms/batch): 4.865 AUC: 0.498 max AUC: 0.538
# >> Training model Ganomaly. Epoch 7/15
#    Avg Run Time (ms/batch): 4.863 AUC: 0.529 max AUC: 0.538
# >> Training model Ganomaly. Epoch 8/15
#    Avg Run Time (ms/batch): 4.862 AUC: 0.520 max AUC: 0.538
# >> Training model Ganomaly. Epoch 9/15
#    Avg Run Time (ms/batch): 4.898 AUC: 0.496 max AUC: 0.538
# >> Training model Ganomaly. Epoch 10/15
#    Avg Run Time (ms/batch): 4.885 AUC: 0.523 max AUC: 0.538
# >> Training model Ganomaly. Epoch 11/15
#    Avg Run Time (ms/batch): 4.917 AUC: 0.539 max AUC: 0.539
#   7% 48/703 [00:02<00:25, 26.05it/s]Reloading d net
# >> Training model Ganomaly. Epoch 12/15
#    Avg Run Time (ms/batch): 4.922 AUC: 0.547 max AUC: 0.547
# >> Training model Ganomaly. Epoch 13/15
#    Avg Run Time (ms/batch): 4.824 AUC: 0.516 max AUC: 0.547
# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.866 AUC: 0.542 max AUC: 0.547
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.872 AUC: 0.513 max AUC: 0.547
# >> Training model Ganomaly.[Done]

import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/Users/dionelisnikolaos/Downloads/GANomaly_image.png')
imgplot = plt.imshow(img)
#plt.pause(2)

img2 = mpimg.imread('/Users/dionelisnikolaos/Downloads/GANomaly_image2.png')
imgplot2 = plt.imshow(img2)
#plt.pause(2)

# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 274.149 AUC: 0.621 max AUC: 0.621
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 284.825 AUC: 0.649 max AUC: 0.649

# https://github.com/samet-akcay/ganomaly
# we use: https://github.com/samet-akcay/ganomaly

# Files already downloaded and verified
# >> Training model Ganomaly.
# 100% 703/703 [20:16<00:00,  1.73s/it]
# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 274.149 AUC: 0.621 max AUC: 0.621
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 284.825 AUC: 0.649 max AUC: 0.649
#   2% 11/703 [00:18<20:00,  1.73s/it]


from keras.datasets import cifar10 # we now load the CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data() # CIFAR-10 dataset

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

# https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3
# use: https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3

# Files already downloaded and verified
# >> Training model Ganomaly. Epoch 1/15
#   0% 1/703 [00:01<19:09,  1.64s/it]   Avg Run Time (ms/batch): 238.385 AUC: 0.587 max AUC: 0.587
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 455.465 AUC: 0.589 max AUC: 0.589
# 100% 702/703 [22:10<00:01,  1.74s/it]
# >> Training model Ganomaly. Epoch 3/15
#    Avg Run Time (ms/batch): 247.068 AUC: 0.647 max AUC: 0.647
# >> Training model Ganomaly. Epoch 4/15
#   0% 3/703 [00:07<30:09,  2.59s/it]   Avg Run Time (ms/batch): 254.772 AUC: 0.596 max AUC: 0.647
#  70% 494/703 [19:20<06:27,  1.86s/it]

import tensorflow as tf # use tensorflow
from keras.datasets import mnist # we use keras

# load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8

import matplotlib.pyplot as plt
plt.imshow(x_train[image_index], cmap='Greys')

plt.pause(2)

#x_train.shape
print(x_train.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#input_shape = (64, 64, 1)
input_shape = (28, 28, 1)

# Ensure that the values are float so that we can get decimal points after division
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

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))

model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

# use Adam, use adaptive momentum
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(x=x_train,y=y_train, epochs=10)

model.evaluate(x_test, y_test)
image_index = 4444

plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.pause(2)

pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())

# 12768/60000 [=====>........................] - ETA: 1:13 - loss: 0.0376 - acc: 0.9880
# 12832/60000 [=====>........................] - ETA: 1:12 - loss: 0.0376 - acc: 0.9880
# 12896/60000 [=====>........................] - ETA: 1:12 - loss: 0.0375 - acc: 0.9881
# 12960/60000 [=====>........................] - ETA: 1:12 - loss: 0.0373 - acc: 0.9881

# >> Training model Ganomaly. Epoch 13/15
#    Avg Run Time (ms/batch): 4.910 AUC: 0.528 max AUC: 0.559
# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.875 AUC: 0.533 max AUC: 0.559
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.830 AUC: 0.531 max AUC: 0.559
# >> Training model Ganomaly.[Done]

# Namespace(anomaly_class='bird', batchsize=64, beta1=0.5, dataroot='', dataset='cifar10',
# device='gpu', display=False, display_id=0, display_port=8097, display_server='http://localhost',
# droplast=True, extralayers=0, gpu_ids=[0], isTrain=True, isize=32, iter=0, load_weights=False, lr=0.0002,
# manualseed=-1, metric='roc', model='ganomaly', name='ganomaly/cifar10', nc=3, ndf=64, ngf=64, ngpu=1, niter=15,
# nz=100, outf='./output', phase='train', print_freq=100, proportion=0.1, resume='', save_image_freq=100,
# save_test_images=False, w_bce=1, w_enc=1, w_rec=50, workers=8)

# Files already downloaded and verified
# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 4.057 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 4.791 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 3/15
#    Avg Run Time (ms/batch): 4.897 AUC: 0.519 max AUC: 0.519
# >> Training model Ganomaly. Epoch 4/15
#    Avg Run Time (ms/batch): 4.792 AUC: 0.502 max AUC: 0.519
# >> Training model Ganomaly. Epoch 5/15
#    Avg Run Time (ms/batch): 4.937 AUC: 0.536 max AUC: 0.536
# >> Training model Ganomaly. Epoch 6/15
#    Avg Run Time (ms/batch): 4.883 AUC: 0.498 max AUC: 0.536
# >> Training model Ganomaly. Epoch 7/15
#    Avg Run Time (ms/batch): 4.960 AUC: 0.503 max AUC: 0.536
# >> Training model Ganomaly. Epoch 8/15
#    Avg Run Time (ms/batch): 4.916 AUC: 0.559 max AUC: 0.559
# >> Training model Ganomaly. Epoch 9/15
#    Avg Run Time (ms/batch): 4.870 AUC: 0.522 max AUC: 0.559
# >> Training model Ganomaly. Epoch 10/15
#    Avg Run Time (ms/batch): 4.898 AUC: 0.539 max AUC: 0.559
#  65% 455/703 [00:16<00:08, 28.19it/s]
# >> Training model Ganomaly. Epoch 11/15
#    Avg Run Time (ms/batch): 4.900 AUC: 0.529 max AUC: 0.559
# >> Training model Ganomaly. Epoch 12/15
#    Avg Run Time (ms/batch): 4.856 AUC: 0.541 max AUC: 0.559
# >> Training model Ganomaly. Epoch 13/15
#    Avg Run Time (ms/batch): 4.910 AUC: 0.528 max AUC: 0.559
# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.875 AUC: 0.533 max AUC: 0.559
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.830 AUC: 0.531 max AUC: 0.559
# >> Training model Ganomaly.[Done]



import sklearn
import sklearn.datasets
import sklearn.datasets.kddcup99

"""KDDCUP 99 dataset: A classic dataset for anomaly detection.
The dataset page is available from UCI Machine Learning Repository"""

import sys
import errno
from gzip import GzipFile

import os
import logging
from io import BytesIO
from os.path import exists, join

try:
    #from urllib2 import urlopen
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/kddcup99.py
dataset_kddcup99 = sklearn.datasets.kddcup99.fetch_kddcup99()

# we now use: use: https://searchcode.com/codesearch/view/115660132/
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/kddcup99.py
# use: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/kddcup99.py
# we use: http://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/sklearn/datasets/kddcup99.py

# dataset_kddcup99
print(dataset_kddcup99)

import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
#dataset = pd.read_csv('kddcup.data')
#dataset = pd.read_csv('kddcup.data_10_percent')

#dataset = pd.read_csv('/Users/dionelisnikolaos/Downloads/kddcup.data')
dataset = pd.read_csv('/Users/dionelisnikolaos/Downloads/kddcup.data_10_percent')
# https://github.com/chadlimedamine/kdd-cup-99-Analysis-machine-learning-python/blob/master/kdd_binary_classification_ANN.py

#change Multi-class to binary-class
dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

#encoding categorical data
y = dataset.iloc[:, 41].values
x = dataset.iloc[:, :-1].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()

x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])

onehotencoder_1 = OneHotEncoder(categorical_features = [1])
x = onehotencoder_1.fit_transform(x).toarray()

onehotencoder_2 = OneHotEncoder(categorical_features = [4])
x = onehotencoder_2.fit_transform(x).toarray()

onehotencoder_3 = OneHotEncoder(categorical_features = [70])
x = onehotencoder_3.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split

# use: model_selection
# we use: sklearn.model_selection
#from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)



# use Keras
import keras

from keras.layers import Dense
from keras.models import Sequential

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu', input_dim = 118))

#Adding a second hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

#Adding a third hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 20)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# https://github.com/chadlimedamine/kdd-cup-99-Analysis-machine-learning-python/blob/master/kdd_binary_classification_ANN.py
# use: https://github.com/chadlimedamine/kdd-cup-99-Analysis-machine-learning-python/blob/master/kdd_binary_classification_ANN.py

# the performance of the classification model
print("the Accuracy is: "+ str((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])))

recall = cm[1,1]/(cm[0,1]+cm[1,1])

print("Recall is : "+ str(recall))
print("False Positive rate: "+ str(cm[1,0]/(cm[0,0]+cm[1,0])))

precision = cm[1,1]/(cm[1,0]+cm[1,1])

print("Precision is: "+ str(precision))
print("F-measure is: "+ str(2*((precision*recall)/(precision+recall))))

from math import log # we use log(.)
print("Entropy is: "+ str(-precision*log(precision)))

# 244720/345814 [====================>.........] - ETA: 55s - loss: 0.0038 - acc: 0.9992
# 244770/345814 [====================>.........] - ETA: 55s - loss: 0.0038 - acc: 0.9992
# 244910/345814 [====================>.........] - ETA: 54s - loss: 0.0038 - acc: 0.9992

import tensorflow as tf
import tensorflow_datasets as tfds

# use: https://www.tensorflow.org/datasets
# we now use: https://www.tensorflow.org/datasets

tf.enable_eager_execution() # tfds works in both Eager and Graph modes
print(tfds.list_builders()) # See the available datasets

# Construct a tf.data.Dataset
dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)

# we now use: https://www.tensorflow.org/datasets
dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

for features in dataset.take(1):
  image, label = features["image"], features["label"]

# https://www.tensorflow.org/datasets
# use: https://www.tensorflow.org/datasets



import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import numpy.random
import scipy.stats as ss
from sklearn.mixture import GaussianMixture

import os
import tensorflow as tf
from sklearn import metrics

from gluoncv.data import ImageNet
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

from gluoncv import data, utils
from matplotlib import pyplot as plt

import scipy.io as sio
import matplotlib.pyplot as plt

image_ind = 10 # define the index
#train_data = sio.loadmat('train_32x32.mat')
train_data = sio.loadmat('/Users/dionelisnikolaos/Downloads/train_32x32.mat')

# SVHN Dataset: Street View House Numbers (SVHN)
# https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2
# use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2

# access to the dict
x_train = train_data['X']
y_train = train_data['y']

# show sample
plt.imshow(x_train[:,:,:,image_ind])
plt.show()

print(y_train[image_ind])
image_ind = 10 # index, image index
test_data = sio.loadmat('/Users/dionelisnikolaos/Downloads/test_32x32.mat')

# access to the dict
x_test = test_data['X']
y_test = test_data['y']

# show sample
plt.imshow(x_test[:,:,:,image_ind])
plt.show()

print(y_test[image_ind])



# The UCI HAR Dataset
DATASET_PATH = "/Users/dionelisnikolaos/Downloads/UCI HAR Dataset/"

TRAIN = "train/"
TEST = "test/"

# Load "X" (the neural network's training and testing inputs)
# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/README.md

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')

        # Read dataset from disk, dealing with text files' syntax
        X_signals.append([np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file]])

        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"]

X_train_signals_paths = [DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
X_test_signals_paths = [DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)

# Load "y" (the neural network's training and testing outputs)
def load_y(y_path):
    file = open(y_path, 'r')

    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32)
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# Input Data
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series

n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

print('')
print(X_train.shape)
print(X_test.shape)

print('')
print(y_train.shape)
print(y_test.shape)

print('')
print(y_train)

print('')
print(y_test)

print('')
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print('')

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print('')

print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print('')



# http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html
# use: http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html

phi_i = 1/7
mu_1 = [0.0, 1.0]
mu_2 = [0.75, 0.6]
mu_3 = [1.0, 0.0]
mu_4 = [0.45, -0.8]
mu_5 = [-0.45, -0.8]
mu_6 = [-0.95, -0.2]
mu_7 = [-0.8, 0.65]

mu_total = [mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7]
sigmaSquared_i = 0.01*np.eye(2)

# we use: http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html
# v_gaussmixp([], [[0, 1]; [0.75, 0.70]; [1, 0]; [0.48, -0.8]; [-0.48, -0.8]; [-1, -0.24]; [-0.8, 0.6]], [[0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]], [1, 1, 1, 1, 1, 1, 1]')
# use: v_gaussmixp([], [[0, 1]; [0.75, 0.70]; [1, 0]; [0.48, -0.8]; [-0.48, -0.8]; [-1, -0.24]; [-0.8, 0.6]], [[0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]], [1, 1, 1, 1, 1, 1, 1]')

# find GMM probability
def prob21(x):
    prob = 0.0

    #print(x)
    x = np.transpose(x)
    #print(np.transpose(x))

    #print(phi_i)
    #print((np.linalg.det(sigmaSquared_i)))

    for i in range(7):
        #prob = prob + (phi_i * ((1 / np.sqrt(((2*np.pi)**7)*(np.linalg.det(sigmaSquared_i)))) * np.exp(-0.5*np.transpose(x-np.transpose(mu_total[i]))*(np.linalg.inv(sigmaSquared_i))*(x-np.transpose(mu_total[i])))))
        #prob = prob + (phi_i * ((1 / np.sqrt(((2*np.pi)**7)*(np.linalg.det(sigmaSquared_i)))) * np.exp(-0.5*(np.transpose(x-np.transpose(mu_total[i])))*(np.linalg.inv(sigmaSquared_i))*((x-np.transpose(mu_total[i]))))))

        #prob = prob + (phi_i * ((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))) * np.exp(-0.5 * ((x - (mu_total[i]))) * (np.linalg.inv(sigmaSquared_i)) * (np.transpose(x - (mu_total[i]))))))
        #print((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))))
        #print(mu_total[i])

        var1 = ((x - (mu_total[i])))
        var1 = np.array(var1)

        #prob = prob + (phi_i * ((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))) * np.exp(
        #    -0.5 * (((var1)) * (np.linalg.inv(sigmaSquared_i)) * ((var1.T))))))

        #prob = prob + (phi_i * ((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))) * np.exp(
        #    -0.5 * (((var1.T).dot((np.linalg.inv(sigmaSquared_i)))).dot(var1)))))

        prob = prob + (phi_i * ((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))) * np.exp(
            -0.5 * (((var1).dot((np.linalg.inv(sigmaSquared_i)))).dot(var1)))))

    return prob

#prob21([1.0, 0.0])
print(prob21([1.0, 0.0]))

# http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html
# we use: http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html
# v_gaussmixp([], [[0, 1]; [0.75, 0.70]; [1, 0]; [0.48, -0.8]; [-0.48, -0.8]; [-1, -0.24]; [-0.8, 0.6]], [[0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]], [1, 1, 1, 1, 1, 1, 1]')
# v_gaussmixp([], [[0, 1]; [0.75, 0.70]; [1, 0]; [0.48, -0.8]; [-0.48, -0.8]; [-1, -0.24]; [-0.8, 0.6]], [[0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]], [1, 1, 1, 1, 1, 1, 1]')

print(prob21([0.0, 1.0]))
print(prob21([0.0, 0.0]))



import numpy as np # numpy
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture

#X = GMMSamples(W, mu, sigma, d)
#gmm = GMM(110, covariance_type='full', random_state=0)

import numpy.random
import scipy.stats as ss

import matplotlib
import matplotlib.pyplot as plt

import os
import tensorflow as tf
from sklearn import metrics

# UCI HAR Dataset, and https://www.tensorflow.org/datasets
DATASET_PATH = "/Users/dionelisnikolaos/Downloads/UCI HAR Dataset/"
# we use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2

TRAIN = "train/"
TEST = "test/"

# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')

        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]])

        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

INPUT_SIGNAL_TYPES = ["body_acc_x_", "body_acc_y_", "body_acc_z_",
                      "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
                      "total_acc_x_", "total_acc_y_", "total_acc_z_"]

# Output classes to learn how to classify
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

X_train_signals_paths = [DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
X_test_signals_paths = [DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)

# Load "y" (the neural network's training and testing outputs)
def load_y(y_path):
    file = open(y_path, 'r')

    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32)

    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# Input Data
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series

n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

print('')
print(X_train.shape)
print(X_test.shape)

print('')
print(y_train.shape)
print(y_test.shape)



# LSTM Neural Network's internal structure
n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)

learning_rate = 0.0025 # For training
lambda_loss_amount = 0.0015 # Training

batch_size = 1500 # we define the batch size
display_iter = 30000  # To show test set accuracy during training
training_iters = training_data_count * 300  # Loop 300 times on the dataset

print('')
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))

print('')
print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")

print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print('')

# we now use the LSTM RNN model
def LSTM_RNN(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.

    # Note, some code of this notebook is inspired from an slightly different RNN architecture used on
    # another dataset, some of the credits goes to "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size

    # Reshape to prepare the input to the hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # Shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) # For the new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)

    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size

    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s

def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes

    # e.g.: one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/README.md
# use: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/README.md



mean = [0, 0] # define the mean
cov = [[1, 0], [0, 100]] # diagonal covariance

import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 5000).T

plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()

n = 10000
numpy.random.seed(0x5eed)

# Parameters of the mixture components
norm_params = np.array([[5, 1], [1, 1.3], [9, 1.3]])
n_components = norm_params.shape[0]

# Weight of each component, in this case all of them are 1/3
weights = np.ones(n_components, dtype=np.float64) / float(n_components)

# A stream of indices from which to choose the component
mixture_idx = numpy.random.choice(n_components, size=n, replace=True, p=weights)

# y is the mixture sample
y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                   dtype=np.float64)

# Theoretical PDF plotting -- generate the x and y plotting positions
xs = np.linspace(y.min(), y.max(), 200)
ys = np.zeros_like(xs)

for (l, s), w in zip(norm_params, weights):
    ys += ss.norm.pdf(xs, loc=l, scale=s) * w

plt.plot(xs, ys)
plt.hist(y, normed=True, bins="fd")

plt.xlabel("x")
plt.ylabel("f(x)")

plt.show()



# Generate synthetic data
N,D = 1000, 2 # number of points and dimenstinality

if D == 2:
    #set gaussian ceters and covariances in 2D
    means = np.array([[0.5, 0.0],
                      [0, 0],
                      [-0.5, -0.5],
                      [-0.8, 0.3]])
    covs = np.array([np.diag([0.01, 0.01]),
                     np.diag([0.025, 0.01]),
                     np.diag([0.01, 0.025]),
                     np.diag([0.01, 0.01])])
elif D == 3:
    # set gaussian ceters and covariances in 3D
    means = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [-0.5, -0.5, -0.5],
                      [-0.8, 0.3, 0.4]])
    covs = np.array([np.diag([0.01, 0.01, 0.03]),
                     np.diag([0.08, 0.01, 0.01]),
                     np.diag([0.01, 0.05, 0.01]),
                     np.diag([0.03, 0.07, 0.01])])

n_gaussians = means.shape[0]

points = []
for i in range(len(means)):
    x = np.random.multivariate_normal(means[i], covs[i], N )
    points.append(x)

points = np.concatenate(points)

gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
gmm.fit(points) # we fit the Gaussian model

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

import os
import matplotlib.cm as cmx
import numpy as np # use numpy

def visualize_3d_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 3D
    Input:
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))

    # Visualize data
    fig = plt.figure(figsize=(8, 8))

    axes = fig.add_subplot(111, projection='3d')

    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set_zlim([-1, 1])

    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))

    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)

        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title('3D GMM')

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    axes.view_init(35.246, 45)

    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/3D_GMM_demonstration.png', dpi=100, format='png')

    plt.show()

def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input:
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    pi = np.pi
    cos = np.cos
    sin = np.sin

    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]

    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]

    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')

    c = cmap.to_rgba(w)
    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)
    return ax

def visualize_2D_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 2D

    Input:
        points: N X 2, sampled points
        w: n_gaussians, gmm weights
        mu: 2 X n_gaussians, gmm means
        stdev: 2 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))

    fig = plt.figure(figsize=(8, 8)) # Visualize
    axes = plt.gca() # Visualize data
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])

    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))

    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        plt.scatter(points[idx, 0], points[idx, 1], alpha=0.3, c=colors[i])

        for j in range(8):
            axes.add_patch(
                patches.Ellipse(mu[:, i], width=(j+1) * stdev[0, i], height=(j+1) *  stdev[1, i], fill=False, color=[0.0, 0.0, 1.0, 1.0/(0.5*j+1)]))

        plt.title('GMM')

    plt.xlabel('X')
    plt.ylabel('Y')

    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/2D_GMM_demonstration.png', dpi=100, format='png')

    plt.show()

#visualize
if D == 2:
    visualize_2D_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
elif D == 3:
    visualize_3d_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)



#Caltech-101 Dataset
#CIFAR-10 Dataset, CIFAR-100 Dataset

from keras.models import Sequential # use Sequential
from keras.layers import Dense # use FC fully connected

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
import scipy.io
import scipy.misc
import numpy as np

import tensorflow as tf # tensorflow
import matplotlib.pyplot as plt # matplotlib

def imread(path):
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2:
		img = np.transpose(np.array([img, img, img]), (2, 0, 1))
	return img

#cwd = os.getcwd()
#path = cwd + "/101_ObjectCategories"

#path = "/101_ObjectCategories"
path = "/Users/dionelisnikolaos/Downloads/101_ObjectCategories"

#CIFAR-10 Dataset and Caltech-101 Dataset
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)

imgs = []
labels = []

print('') #print(categories)
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

import pandas as pd # use pandas
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

# CIFAR-10 dataset
import numpy # numpy
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

from keras import backend as K
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)
K.set_image_dim_ordering('th')

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1] # we now create the model
# we use: https://github.com/acht7111020/CNN_object_classification

model = Sequential() # we now use Sequential
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



# Caltech-101 Dataset
# CIFAR-10 and CIFAR-100 Datasets

# we use now Sequential
from keras.models import Sequential
from keras.layers import Dense

# use dropout
from keras.layers import Dropout
from keras.layers import Flatten

from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras import initializers
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('th')

import os
import scipy.io
import scipy.misc
import numpy as np

import tensorflow as tf # tensorflow
import matplotlib.pyplot as plt # matplotlib

def imread(path):
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2:
		img = np.transpose(np.array([img, img, img]), (2, 0, 1))
	return img

#cwd = os.getcwd()
#path = "/101_ObjectCategories"
#path = cwd + "/101_ObjectCategories"
path = "/Users/dionelisnikolaos/Downloads/101_ObjectCategories"

# CIFAR-10 Dataset
# The Caltech-101 Dataset
# CIFAR-10 Dataset, and CIFAR-100 Dataset

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

import pandas as pd # use pandas
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.1)

X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)

X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)

print("Num train_imgs: %d" % (len(X_train)))
print("Num test_imgs: %d" % (len(X_test)))

# One hot encode outputs
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

epochs = 300 # compile the model
lrate = 0.0001 # we compile the model

decay = lrate / epochs

adam = SGD(lr=0.0001)
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())
np.random.seed(seed)

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
				 epochs=epochs, batch_size=56, shuffle=True, callbacks=[earlyStopping])

# The final evaluation of the model
# hist = model.load_weights('./64.15/model.h5')
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

#import sklearn.datasets
#import sklearn.datasets2
#dataset_boston = datasets.load_boston()
#dataset_boston = datasets2.load_boston()

#dataset_kddcup99 = datasets2.load_digits()



import scipy.io # we use .io
#mat2 = scipy.io.loadmat('NATOPS6.mat')
mat2 = scipy.io.loadmat('/Users/dionelisnikolaos/Downloads/NATOPS6.mat')

print(mat2) # use NATOPS6.mat
#mat = scipy.io.loadmat('thyroid.mat')
mat = scipy.io.loadmat('/Users/dionelisnikolaos/Downloads/thyroid.mat')

# thyroid.mat
print(mat)



# usenet_recurrent3.3.data
# we use: usenet_recurrent3.3.data

import numpy # we use numpy
import pandas as pd # use pandas
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

from csv import reader
# we use: https://skymind.ai/wiki/open-datasets
# use: http://people.csail.mit.edu/yalesong/cvpr12/

# Load a CSV file
def load_csv(filename):
    file = open(filename, "r")

    lines = reader(file)
    dataset = list(lines)

    return dataset

dataset = load_csv('/Users/dionelisnikolaos/Downloads/ann-train.data.txt')

filename = '/Users/dionelisnikolaos/Downloads/ann-train.data.txt' # Load the dataset
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

        # If you need to use the file content as numbers:
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

#data_dir="./datasets/KDD-CUP-99/" # or: data_dir="./"

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

import os
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.callbacks import CallbackList, ModelCheckpoint
from keras.regularizers import l2

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#from keras.applications.inception_v3 import InceptionV3
#base_model = InceptionV3(weights='imagenet', include_top=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

num_train_images =  1500
num_test_images = 100

import os
import h5py

import glob
import numpy as np

# we use opencv-python
import cv2

# we use keras
from keras.preprocessing import image

# dataset pre-processing
#train_path   = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\train"
#test_path    = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test"

#train_path   = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\train"
train_path   = "/Users/dionelisnikolaos/Downloads/dataset/train"

#test_path    = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test"
test_path    = "/Users/dionelisnikolaos/Downloads/dataset/test"

train_labels = os.listdir(train_path)
test_labels  = os.listdir(test_path)

image_size       = (64, 64)
num_train_images = 1500
num_test_images  = 100

# tunable parameters
num_channels     = 3

# train_x dimension = {(64*64*3), 1500}
# train_y dimension = {1, 1500}
# test_x dimension  = {(64*64*3), 100}
# test_y dimension  = {1, 100}

train_x = np.zeros(((image_size[0]*image_size[1]*num_channels), num_train_images))
train_y = np.zeros((1, num_train_images))

test_x  = np.zeros(((image_size[0]*image_size[1]*num_channels), num_test_images))
test_y  = np.zeros((1, num_test_images))

count = 0 # The TRAIN dataset
num_label = 0 # For the TRAIN dataset

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

# The TEST dataset
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

# Standardization
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

# save using h5py
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

	params = {"w": w, "b": b} # param dict
	grads  = {"dw": dw, "db": db} # gradient dict

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

# here, we now use: https://gogul09.github.io/software/neural-nets-logistic-regression
epochs = 10 # https://gogul09.github.io/software/neural-nets-logistic-regression
lr = 0.0003 # define the learning rate, lr, step size

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
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ADAM, adaptive momentum
# we use the Adam optimizer

# we now fit the model
#model.fit(x=x_train,y=y_train, epochs=10)

#model.fit(x=x_train,y=y_train, epochs=10)
model.fit(x=x_train,y=y_train, epochs=8)

model.evaluate(x_test, y_test) # we evaluate the model
# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

# use index 4444
image_index = 4444

plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.pause(2) #plt.pause(5)

#pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))

print(pred.argmax())

# Deep Generative Models
# GANs and VAEs, Generative Models

# We use batch normalisation.
# Random noise: From random noise to a tensor
# GANs are very difficult to train. Super-deep models. This is why we use batch normalisation.

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb



# Anomaly detection (AD)
# GANs for super-resolution
# Unsupervised machine learning
# Generative Adversarial Networks, GANs

# the BigGAN dataset
# BigGAN => massive dataset
# latent space, BigGAN, GANs

# down-sampling, sub-sample, pooling
# throw away samples, pooling, max-pooling
# partial derivatives, loss function and partial derivatives

# https://github.com/Students-for-AI/The-Academy-of-AI
# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models

# Generator G and Discriminator D
# the loss function of the Generator G

# up-convolution
# We use a filter we do up-convolution with.

# use batch normalisation
# GANs are very difficult to train and this is why we use batch normalisation.

# the ReLU activation function
# We normalize across a batch.
# ReLU is the most common activation function. We use ReLU.
# Mean across a batch. We use batches. Normalize across a batch.

# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb



import torch
import torchvision

from torchvision import datasets, transforms
#from torchvision import transforms, datasets

# use matplotlib
import matplotlib.pyplot as plt

# use nn.functional
import torch.nn.functional as F

#import matplotlib.pyplot as plt
#batch_size = 128

# download the training dataset
#train_data = datasets.FashionMNIST(root='fashiondata/',
#                                   transform=transforms.ToTensor(), train=True, download=True)

# we create the train data loader
#train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

# we now define the batch size
batch_size = 100 # define the batch size

train_data = datasets.FashionMNIST(root='fashiondata/',
                                 transform=transforms.ToTensor(), train=True, download=True)

train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size, shuffle=True)

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs
# GANs and LSTM RNNs: Use LSTM RNNs together with GANs

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

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb
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

                plt.pause(5) # pause for some seconds
                g.train() # now, go back to the training mode

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

# Epoch:  0 Batch idx: 0 	Disciminator cost:  1.3832124471664429 	Generator cost:  0.006555716972798109
# Epoch:  0 Batch idx: 1 	Disciminator cost:  1.0811840295791626 	Generator cost:  0.008780254982411861
# Epoch:  0 Batch idx: 2 	Disciminator cost:  0.8481155633926392 	Generator cost:  0.011281056329607964
# Epoch:  0 Batch idx: 3 	Disciminator cost:  0.6556042432785034 	Generator cost:  0.013879001140594482
# Epoch:  0 Batch idx: 4 	Disciminator cost:  0.5069876909255981 	Generator cost:  0.016225570812821388
# Epoch:  0 Batch idx: 5 	Disciminator cost:  0.4130948781967163 	Generator cost:  0.018286770209670067

# Epoch:  0 Batch idx: 41 	Disciminator cost:  0.10074597597122192 	Generator cost:  0.03721988573670387
# Epoch:  0 Batch idx: 42 	Disciminator cost:  0.07906078547239304 	Generator cost:  0.04363853484392166

# Epoch:  0 Batch idx: 118 	Disciminator cost:  0.010556117631494999 	Generator cost:  0.06929603219032288
# Epoch:  0 Batch idx: 119 	Disciminator cost:  0.017774969339370728 	Generator cost:  0.07270769774913788

# Epoch:  0 Batch idx: 447 	Disciminator cost:  0.12328958511352539 	Generator cost:  0.03817861154675484
# Epoch:  0 Batch idx: 448 	Disciminator cost:  0.06865841150283813 	Generator cost:  0.03938257694244385

# generate random latent variable to generate images
z = torch.randn(batch_size, 128)

# generate images, use "forward(.)"
im = g.forward(z) # we use "forward(.)"

plt.imshow(im)

