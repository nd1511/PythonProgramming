# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

from __future__ import print_function

# we use LIBROSA for speech processing
import librosa

# 1. Get the file path to the included audio example
# filename = librosa.util.example_audio_file()

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
# y, sr = librosa.load(filename)

# 3. Run the default beat tracker
# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
# beat_times = librosa.frames_to_time(beat_frames, sr=sr)

#print('Saving output to beat_times.csv')
# librosa.output.times_csv('beat_times.csv', beat_times)

import numpy as np

x = np.array(12)
print(x)
print(x.ndim)

x = np.array([12, 3, 6, 14])
print(x)
print(x.ndim)

x = np.array([[12, 3, 6, 14],
             [5, 78, 34, 0]])
print(x)
print(x.ndim)



# the MNIST dataset
from keras.datasets import mnist
# handwritten digit recognition, MNIST

# we use tuples, (..., ...)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('')
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)

#plt.show()
#plt.close()

my_slice = train_images[10:100]

print('')
print(my_slice.shape)

# use imdb
from keras.datasets import imdb

# we use tuples, (..., ...)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print('')
print(train_data.shape)
print(test_data.shape)

print('')
print(train_data[0])
print(train_labels[0])



# we use the CHiME Challenge
# we use: http://spandh.dcs.shef.ac.uk/chime_challenge/data.html

# in MATLAB, we have:
# [y, fs] = readwav('/Volumes/Maxtor/CHiME5/audio/train/S03_U01.CH1.wav');
# size(y), fs
# %soundsc(y, fs)
# %clear sound
# figure; plot((1:length(y))*(1/fs)/(60*60), y); axisenlarge; figbolden; xlabel('Time (h)'); figbolden; ylabel('Amplitude'); figbolden;

import scipy.io.wavfile

# we use "S03_U01.CH1.wav" from the CHiME Challenge
sample_rate, signal = scipy.io.wavfile.read('/Volumes/Maxtor/CHiME5/audio/train/S03_U01.CH2.wav')

print('')
print(sample_rate)
print(signal.shape)

from sklearn import datasets

# use numpy
import numpy as np

# we use the iris dataset
iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target

#print('Class Labels', y)
print('Class Labels', np.unique(y))

# use sklearn
from sklearn.model_selection import train_test_split

# we split the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# we use: test_size=0.3
# the test set is 30% of the data, the training set is 70% of the data

print("Labels counts in y:", np.bincount(y))

print("Labels counts in y_train:", np.bincount(y_train))

print("Labels counts in y_test:", np.bincount(y_test))



# we use: http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

# we use Keras and TensorFlow
# we use the book: Deep Learning with Python by Francois Chollet

# we use: https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/

import numpy as np

import scipy.io.wavfile
from scipy.fftpack import dct

# we use the TIMIT dataset for clean speech

#sample_rate, signal = scipy.io.wavfile.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV')
#sample_rate, signal = scipy.io.wavfile.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TIMIT/TRAIN/DR1/FCJF0/wavSA1')

# we use "wavSA2.wav" which originates from SA2.WAV from TIMIT
#sample_rate, signal = scipy.io.wavfile.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TIMIT/TRAIN/DR1/FCJF0/wavSA2')

# we use "wavSA2.wav" which originates from SA2.WAV
# in MATLAB, we have to use VOICEBOX: http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/voicebox.html
# we use: [y,fs]=readsph('./TIMIT/TIMIT/TRAIN/DR1/FCJF0/SA2.WAV') and writewav(y,fs,'./TIMIT/TIMIT/TRAIN/DR1/FCJF0/wavSA2')

# we use "wavSI648.wav" which originates from SI648.WAV from TIMIT
sample_rate, signal = scipy.io.wavfile.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TIMIT/TRAIN/DR1/FCJF0/wavSI648')

# we use "wavSI648.wav" which originates from SI648.WAV
# in MATLAB, we have to use VOICEBOX: http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/voicebox.html
# we use: [y,fs]=readsph('./TIMIT/TIMIT/TRAIN/DR1/FCJF0/SI648.WAV') and writewav(y,fs,'./TIMIT/TIMIT/TRAIN/DR1/FCJF0/wavSI648')

# we use "wavSI648.wav" which originates from SI648.WAV
# in MATLAB, we have to use VOICEBOX: http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/voicebox.html



# we keep the first 3.5 seconds
#signal = signal[0:int(3.5 * sample_rate)]

pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

frame_size = 0.025
frame_stride = 0.01

# we convert from seconds to samples
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate

signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))

num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length

z = numpy.zeros((pad_signal_length - signal_length))

pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

frames = pad_signal[indices.astype(numpy.int32, copy=False)]



frames *= numpy.hamming(frame_length)
# frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

NFFT = 512

mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT

pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

nfilt = 40

low_freq_mel = 0
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel

mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale

hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))

for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

filter_banks = numpy.dot(pow_frames, fbank.T)

filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability

filter_banks = 20 * numpy.log10(filter_banks)  # dB

num_ceps = 12

mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13


(nframes, ncoeff) = mfcc.shape

n = numpy.arange(ncoeff)

cep_lifter = 22
#cep_lifter = len(mfcc)

lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)

mfcc *= lift  #*

filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)

filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)



# we use speechpy
# we use: https://github.com/astorfi/speechpy

import scipy.io.wavfile as wav
import numpy as np

import speechpy

import speechpy
#import os

#file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav')
file_name = '/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TIMIT/TRAIN/DR1/FCJF0/wavSI648'

fs, signal = wav.read(file_name)

#signal = signal[:,0]

# Example of pre-emphasizing.
signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

# Example of staching frames
frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),
         zero_padding=True)

print('')

# Example of extracting power spectrum
power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)

print('power spectrum shape=', power_spectrum.shape)

# MFCC features
# extract MFCC features

############# Extract MFCC features #############
mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)

print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
print('mfcc feature cube shape=', mfcc_feature_cube.shape)

# log-energy features
# extract log-energy features

############# Extract log-energy features #############
logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)

print('logenergy features=', logenergy.shape)

# we use the CHiME Challenge
# we use: http://spandh.dcs.shef.ac.uk/chime_challenge/data.html

# we use audio data from the CHiME Challenge

# in MATLAB, we have:
# [y, fs] = readwav('/Volumes/Maxtor/CHiME5/audio/train/S03_U01.CH1.wav');
# size(y), fs
# %soundsc(y, fs)
# %clear sound
# %figure; plot((1:length(y))*(1/fs), y); axisenlarge; figbolden; xlabel('Time (s)'); figbolden; ylabel('Amplitude'); figbolden;
# %figure; plot((1:length(y))*(1/fs)/60, y); axisenlarge; figbolden; xlabel('Time (m)'); figbolden; ylabel('Amplitude'); figbolden;
# figure; plot((1:length(y))*(1/fs)/(60*60), y); axisenlarge; figbolden; xlabel('Time (h)'); figbolden; ylabel('Amplitude'); figbolden;

# we use "S03_U01.CH1.wav" from the CHiME Challenge
sample_rate, signal = scipy.io.wavfile.read('/Volumes/Maxtor/CHiME5/audio/train/S03_U01.CH1.wav')

# CHiME Challenge
# audio data from the CHiME Challenge

print('')
print(sample_rate)
print(signal.shape)

# we use TensorFlow that allows us to express any computation as a graph of data flows
# the nodes in the graph of data flows represent mathematical operations
# the edges in the graph of data flows represent data



# Deep Generative Models
# GANs and VAEs, Generative Models

# random noise
# from random noise to a tensor

# We use batch normalisation.
# GANs are very difficult to train. Super-deep models. This is why we use batch normalisation.

# Anomaly detection (AD)
# Unsupervised machine learning

# GANs for super-resolution
# Generative Adversarial Networks, GANs

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

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

# use PyTorch
import torch

import torch
import torchvision

from torchvision import datasets, transforms

# use matplotlib
import matplotlib.pyplot as plt

batch_size = 128

# download the training dataset
train_data = datasets.FashionMNIST(root='fashiondata/',
                                   transform=transforms.ToTensor(),
                                   train=True,
                                   download=True)

# we create the train data loader
train_loader = torch.utils.data.DataLoader(train_data,
                                           shuffle=True,
                                           batch_size=batch_size)

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# class for D and G
# we train the discriminator and the generator

# we make the discriminator
class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # one-channel, stride of 2
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        # do 1D convolution

        # do 2D convolution
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        # fully connected fc
        self.fc = torch.nn.Linear(128*7*7, 1)
        # the output is a single number, one number

        # we need fc
        # we need a fully connected layer

        # batch normalisation layer
        self.bn1 = torch.nn.BatchNorm2d(64)
        # after the 1D convolution

        # second batch normalization layer
        self.bn2 = torch.nn.BatchNorm2d(128)
        # after the 2D convolution

        # activation function
        #self.af = torch.nn.Sigmoid()
        self.af = torch.nn.ReLU()

        # for the output
        self.s = torch.nn.Sigmoid()

    def forward(selfself, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.af(x)
        x = self.conv2(x)

        x = self.bn2(x)
        x = self.af(x)

        # reshape
        x = x.view(-1, 128*7*7)
        # we do not care about the rows, hence "-1"

        # we do not care about the batch size
        # we do not care about the rows, hence "-1"

        # fully connected (fc)
        x = self.fc(x)

        x = self.s(x)

        return x

# We normalize across a batch.
# Mean across a batch. We use batches. Normalize across a batch.

# use batch normalisation
# GANs are very difficult to train and this is why we use batch normalisation.

# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# this was for the discriminator
# we now do the same for the generator

# Generator G
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # random noise
        # create random noise

        # 128 to 1256
        self.dense1 = torch.nn.Linear(128, 256)

        self.dense2 = torch.nn.Linear(256, 1024)
        self.dense3 = torch.nn.Linear(1024, 128*7*7)

        # convolution layer
        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # we use a stride of 2

        # second convolution layer
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, 4, 2, 1)

        # batch normalization
        self.bn1 = torch.nn.BatchNorm1d(256)

        # second batch normalization layer
        self.bn2 = torch.nn.BatchNorm1d(1024)
        # this is after dense2

        # this is after dense3
        self.bn3 = torch.nn.BatchNorm1d(128*7*7)

        self.bn4 = torch.nn.BatchNorm2d(64)

        # use ReLU
        self.af = torch.nn.ReLU()

        self.s = torch.nn.Sigmoid()

        # grayscale images
        # we use grayscale images

    # forward function
    def forward(self, z):
        #z = self.dense1(z)
        #z = self.bn1(z)
        #z = self.af(z)

        z = self.af(self.bn1(self.dense1(z)))

        #z = self.dense2(z)
        #z = self.bn2(z)
        #z = self.af(z)

        z = self.af(self.bn2(self.dense2(z)))

        z = self.af(self.bn3(self.dense3(z)))

        # up-convolution
        z = self.af(self.bn4(self.uconv1(z)))

        # stable training
        # batch normalization for stable training

        z = self.s(self.uconv2(z))

        return z

# this was for the generator and the discriminator
# we do the same for the generator and the discriminator

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb
# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

d = Discriminator()
g = Generator()

epochs = 100

dlr = 0.0003
glr = 0.0003

#d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)
#g_optimizer = torch.

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# instantiate the model
d = Discriminator()
g = Generator()

# training hyperparameters
epochs = 100

# training hyperparameters
dlr = 0.0003
glr = 0.0003

# we use Adam
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

def train(epochs):
    for epoch in range(epochs):
        #for batch_idx, (real_images, _) enumerate(train_loader):

        for batch_idx, (real_images, _) in enumerate(train_loader):
            # random noise
            z = torch.randn(batch_size, 128)

            # latent space
            # our latent space is 128

            # generate images
            generated_images = g(z)

            gen_pred = d(generated_images)

            real_pred = d(real_images)

            # cost function
            # create loss function

            # sum over batches
            #dcost = -torch.sum(torch.log(real_pred))

            dcost = -torch.sum(torch.log(real_pred)) - torch.sum(torch.log(1 - real_pred))

            # we sum over the batches
            gcost = -torch.sum(torch.log(gen_pred)) / batch_size
            # use: . / batch_size

            d_optimizer.zero_grad()

            # delete stuff from the computational graph
            dcost.backward(retain_graph=True)

            d_optimizer.step()

            g_optimizer.zero_grad()
            gcost.backward()

            g_optimizer.step()

            # batch normalization
            # different between training and testing

            # batch normalization is different between training and testing

            # running average during testing
            # we use the running average during testing

            if batch_idx == 10000:
                # batch normalization is different between training and testing
                g.eval()

                noise_input = torch.randn(1,128)
                generated_image = g(noise_input)

                # use .squeeze()
                generated_img.imshow(generated_image.detach().squeeze())

                # batch normalization is different between training and testing
                g.train()

            dcost /= batch_size
            gcost /= batch_size

            # for every epoch, print
            print('Epoch:', epoch, '\tBatch:', batch_idx)

            dcosts.append(dcost.item())
            gcosts.append(gcost.item())

            loss_ax.plot(dcosts, 'r')
            loss_ax.plot(gcosts, 'b')

            fig.canvas.draw()

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

train(epochs)

