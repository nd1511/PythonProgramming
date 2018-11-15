#####################################################################################
# Training the CNN for cepstral domain approach III.
# Input:
#       1- Training input: Train_inputSet_g711.mat
#       2- Training target: Train_targetSet_g711.mat
#       3- Validation input: Validation_inputSet_g711.mat
#       4- Validation target: Validation_targetSet_g711.mat
# Output:
#       1- Trained CNN weights: cnn_weights_ceps_g711_best_example.h5
#####################################################################################


import os
import tensorflow as tf
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Add, Multiply, Average, Activation, LeakyReLU
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D
from keras import backend as K
import keras.optimizers as optimizers
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
import keras.callbacks as cbs
from numpy import random
import scipy.io as sio
from sklearn import preprocessing
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
codec = "g711"
fram_length = 32
n1 = 22  #  F=22 in paper
n2 = 44  #
n3 = 22  #
N_cnn = 6 # N=6 in paper

# Training parameters
nb_epochs = 2
batch_size = 16
learning_rate = 5e-4

#####################################################################################
# 1. load data
#####################################################################################

print('> Loading data... ')
# Load Input Data
mat_input = "./data/Train_inputSet_g711.mat"
mat_input = os.path.normcase(mat_input)
x_train_noisy = sio.loadmat(mat_input)
x_train_noisy = x_train_noisy['inputSetNorm']
x_train_noisy = np.array(x_train_noisy)

# Load Input Data for Validation
mat_input_vali = "./data/Validation_inputSet_g711.mat"
mat_input_vali = os.path.normcase(mat_input_vali)
x_train_noisy_vali = sio.loadmat(mat_input_vali)
x_train_noisy_vali = x_train_noisy_vali['inputSetNorm']
x_train_noisy_vali = np.array(x_train_noisy_vali)

# Load Target Data
mat_target = "./data/Train_targetSet_g711.mat"
mat_target = os.path.normcase(mat_target)
x_train = sio.loadmat(mat_target)
x_train = x_train['targetSet']
x_train = np.array(x_train)

# Load Target Data for Validation
mat_target_vali = "./data/Validation_targetSet_g711.mat"
mat_target_vali = os.path.normcase(mat_target_vali)
x_train_vali = sio.loadmat(mat_target_vali)
x_train_vali = x_train_vali['targetSet']
x_train_vali = np.array(x_train_vali)

# Randomization of Training Pairs
random.seed(1024)
train = np.column_stack((x_train_noisy, x_train))
np.random.shuffle(train)
x_train_noisy = train[:, :fram_length]
x_train = train[:, fram_length:]

# Reshape of Traing Pairs and validation Pairs
x_train_noisy = np.reshape(x_train_noisy, (x_train_noisy.shape[0], x_train_noisy.shape[1], 1))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train_noisy_vali = np.reshape(x_train_noisy_vali, (x_train_noisy_vali.shape[0], x_train_noisy_vali.shape[1], 1))
x_train_vali = np.reshape(x_train_vali, (x_train_vali.shape[0], x_train_vali.shape[1], 1))

print('> Data Loaded. Compiling...')

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

adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='mse', metrics=[snr])

#####################################################################################
# 3. Fit the model
#####################################################################################

# Stop criteria
stop_str = cbs.EarlyStopping(monitor='val_loss', patience=16, verbose=1, mode='min')
# Reduce learning rate
reduce_LR = cbs.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0)
# Save only best weights
best_weights = "./data/cnn_weights_ceps_g711_example.h5"
best_weights = os.path.normcase(best_weights)
model_save = cbs.ModelCheckpoint(best_weights, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True, period=1)

start = time.time()
print("> Training model " + "using Batch-size: " + str(batch_size) + ", Learning_rate: " + str(learning_rate) + "...")
hist = model.fit(x_train_noisy, x_train, epochs=nb_epochs, batch_size=batch_size, shuffle=True, initial_epoch=0,
                      callbacks=[reduce_LR, stop_str, model_save],
                      validation_data=[x_train_noisy_vali, x_train_vali]
                      )

print("> Saving Completed, Time : ", time.time() - start)
print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')



