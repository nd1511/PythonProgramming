#####################################################################################
# Use the trained CNN for cepstral domain approach III.
# Input:
#       1- CNN input: type_3_cnn_input_ceps_v73.mat
#       2- Trained CNN weights: cnn_weights_ceps_g711_best.h5
# Output:
#       1- CNN output: type_3_cnn_output_ceps.mat
#####################################################################################


import os
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import Input, Add, Multiply, Average, Activation, LeakyReLU
from keras.layers import merge, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler
import keras.optimizers as optimizers
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py as h5
import scipy.io.wavfile as swave
from sklearn import preprocessing
from weightnorm import AdamWithWeightnorm
from tensorflow.python.framework import ops
import math
import time


def snr(y_true, y_pred):
    """
        SNR is Signal to Noise Ratio
    """
    return 10.0 * K.log((K.sum(K.square(y_true))) / (K.sum(K.square(y_pred - y_true)))) / K.log(10.0)

#####################################################################################
# 0. Setup
#####################################################################################

# Settings and CNN topology parameters
fram_length = 32
n1 = 22  #  F=22 in paper
n2 = 44  #
n3 = 22  #
N_cnn = 6 # N=6 in paper

#####################################################################################
# 2. define model
#####################################################################################

input_vec = Input(shape=(fram_length, 1))
c1 = Conv1D(n1, N_cnn, padding='same')(input_vec)
c1 = LeakyReLU(0.2)(c1)
c1 = Conv1D(n1, N_cnn, padding='same')(c1)
c1 = LeakyReLU(0.2)(c1)
x = MaxPooling1D(2)(c1)

c2 = Conv1D(n2, N_cnn, padding='same')(x)
c2 = LeakyReLU(0.2)(c2)
c2 = Conv1D(n2, N_cnn, padding='same')(c2)
c2 = LeakyReLU(0.2)(c2)
x = MaxPooling1D(2)(c2)

c3 = Conv1D(n3, N_cnn, padding='same')(x)
c3 = LeakyReLU(0.2)(c3)
x = UpSampling1D(2)(c3)

c2_2 = Conv1D(n2, N_cnn, padding='same')(x)
c2_2 = LeakyReLU(0.2)(c2_2)
c2_2 = Conv1D(n2, N_cnn, padding='same')(c2_2)
c2_2 = LeakyReLU(0.2)(c2_2)

m1 = Add()([c2, c2_2])
m1 = UpSampling1D(2)(m1)

c1_2 = Conv1D(n1, N_cnn, padding='same')(m1)
c1_2 = LeakyReLU(0.2)(c1_2)
c1_2 = Conv1D(n1, N_cnn, padding='same')(c1_2)
c1_2 = LeakyReLU(0.2)(c1_2)

m2 = Add()([c1, c1_2])

decoded = Conv1D(1, N_cnn, padding='same', activation='linear')(m2)
model = Model(input_vec, decoded)
model.summary()

model.load_weights("./data/cnn_weights_ceps_g711_best.h5")

#####################################################################################
# 4. Test
#####################################################################################

print('> Loading Test data ... ')

mat_input = "./data/type_3_cnn_input_ceps_v73.mat"
mat_input = os.path.normcase(mat_input)

x_test_noisy = h5.File(mat_input, 'r')
x_test_noisy = x_test_noisy.get('inputTestNorm')
x_test_noisy = np.array(x_test_noisy)
x_test_noisy = np.transpose(x_test_noisy)
x_test_noisy = np.reshape(x_test_noisy,(x_test_noisy.shape[0], x_test_noisy.shape[1], 1))

predicted = model.predict(x_test_noisy)

preOutput = "./data/type_3_cnn_output_ceps.mat"
preOutput = os.path.normcase(preOutput)

sio.savemat(preOutput, {'predictions': predicted})


