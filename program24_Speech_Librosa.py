
# we use speech, deep learning (DL), neural networks (NNs) with speech
# we use Li Deng's book: http://125.234.102.146:8080/dspace/bitstream/DNULIB_52011/6853/1/automatic_speech_recognition_a_deep_learning_approach.pdf

# use: https://github.com/Owen864720655/E-book/blob/master/New%20Era%20for%20Robust%20Speech%20Recognition-Exploiting%20Deeping%20Learning.pdf
# use: http://125.234.102.146:8080/dspace/bitstream/DNULIB_52011/6853/1/automatic_speech_recognition_a_deep_learning_approach.pdf

# book for speech: https://github.com/Owen864720655/E-book/blob/master/New%20Era%20for%20Robust%20Speech%20Recognition-Exploiting%20Deeping%20Learning.pdf



# use: https://lidengsite.wordpress.com/book-chapters/
# we use: http://125.234.102.146:8080/dspace/bitstream/DNULIB_52011/6853/1/automatic_speech_recognition_a_deep_learning_approach.pdf

# we also use: https://lidengsite.files.wordpress.com/2018/03/chapter-1.pdf

# we use the TIMIT database
# we use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2

# use: https://github.com/librosa/tutorial/blob/master/Librosa%20tutorial.ipynb

# we use: https://github.com/keunwoochoi/kapre/blob/master/examples/example_codes.ipynb
# we also use: https://github.com/keunwoochoi/kapre/blob/master/examples/prepare%20audio.ipynb

import numpy as np

import keras
import kapre

# use Kapre as a Keras pre-processor
import kapre
# use: https://github.com/keunwoochoi/kapre

# we use LIBROSA, Columbia University, Dan Ellis
import librosa
from librosa import display

# we use plt to plot figures
import matplotlib.pyplot as plt

# we use the TIMIT clean speech database
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA1', sr=None)
#print(len(y), sr)

print(librosa.samples_to_time(len(y), sr))

D = librosa.stft(y)
#print(D.shape, D.dtype)

S, phase = librosa.magphase(D)
#print(S.dtype, phase.dtype, np.allclose(D, S * phase))

plt.figure(figsize=(14, 4))

logPowerSpectrum = np.log(np.abs(librosa.stft(y, 512, 256)) ** 2)

#display.specshow(np.log(np.abs(librosa.stft(y, 512, 256)) ** 2), y_axis='linear', sr=sr)
#display.specshow(logPowerSpectrum, y_axis='linear', x_axis='time', sr=sr)

display.specshow(logPowerSpectrum, y_axis='linear', sr=sr)

plt.title('Log-Spectrogram')
plt.show()



y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA2', sr=None)
#print(len(y), sr)

print(librosa.samples_to_time(len(y), sr))

D = librosa.stft(y)
#print(D.shape, D.dtype)

S, phase = librosa.magphase(D)
#print(S.dtype, phase.dtype, np.allclose(D, S * phase))

plt.figure(figsize=(14, 4))

#logPowerSpectrum = np.log(np.abs(librosa.stft(y, 512, 256)) ** 2)

# we use: https://librosa.github.io/librosa/generated/librosa.core.stft.html
#logPowerSpectrum = np.log(np.abs(librosa.stft(y, n_fft=512, hop_length=256)) ** 2)

logPowerSpectrum = np.log(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)) ** 2)

#display.specshow(np.log(np.abs(librosa.stft(y, 512, 256)) ** 2), y_axis='linear', sr=sr)
#display.specshow(logPowerSpectrum, y_axis='linear', sr=sr)

#display.specshow(logPowerSpectrum, y_axis='linear', x_axis='time', sr=sr)
display.specshow(logPowerSpectrum, y_axis='linear', sr=sr)

plt.title('Log-Spectrogram')
plt.show()

plt.figure(figsize=(14, 4))

#librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
#                                                 ref=np.max), y_axis='log', x_axis='time')

#librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
#                                                 ref=np.max), y_axis='linear', x_axis='time')

librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
                                                 ref=np.max), y_axis='linear')

plt.title('Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()



y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSI648', sr=None)
#print(len(y), sr)

print(librosa.samples_to_time(len(y), sr))

D = librosa.stft(y)
#print(D.shape, D.dtype)

S, phase = librosa.magphase(D)
#print(S.dtype, phase.dtype, np.allclose(D, S * phase))

plt.figure(figsize=(14, 4))

#librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
#                                                 ref=np.max), y_axis='log', x_axis='time')

#librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
#                                                 ref=np.max), y_axis='linear', x_axis='time')

librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
                                                 ref=np.max), y_axis='linear')

plt.title('Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()



# for deep learning (DL): http://125.234.102.146:8080/dspace/bitstream/DNULIB_52011/6853/1/automatic_speech_recognition_a_deep_learning_approach.pdf

# we use: https://github.com/Imperial-College-Data-Science-Society/Neural-Networks/blob/master/slides/L2.Neural-Networks.pdf
# use: https://github.com/Imperial-College-Data-Science-Society/Neural-Networks

# for MATLAB: https://github.com/dustinstansbury/medal
# we use: https://github.com/PhDP/mlbop/tree/master/MATLAB-18

# for DL in MATLAB: https://github.com/dustinstansbury/medal

from __future__ import print_function

import librosa
import librosa.display

import matplotlib.pyplot as plt

filename = "./songs/Song1.mp3"
y, sr = librosa.load(filename, sr=22050)

rmse = librosa.feature.rmse(y=y, frame_length=2048*8, hop_length=512*4)
R = librosa.segment.recurrence_matrix(rmse)

plt.figure(figsize=(8, 8))
librosa.display.specshow(R, x_axis='time', y_axis='time')

plt.title('Similarity Matrix')
plt.show()



# we now use: https://stsievert.com/blog/2015/09/01/matlab-to-python/

# we import pylab
from pylab import *

# matrix multiplication
A = rand(3, 3)
A[0:2, 1] = 4

I = A @ inv(A)
I = A.dot(inv(A))

# vector manipulations
t = linspace(0, 4, num=1e3)

y1 = cos(t/2) * exp(-t)
y2 = cos(t/2) * exp(-5*t)

# plotting
figure()
plot(t, y1, label='Slow decay')
plot(t, y2, label='Fast decay')

legend(loc='best')
show()

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat # fyi
from numpy.random import rand, randn  # fyi (or uniform, gaussian_normal)

x = np.linspace(0, 1, num=100)
y = np.exp(-x) * np.cos(2 * np.pi * x)

plt.figure()
plt.plot(x, y, label='Moderate decay')

plt.legend(loc='best')
plt.show()



# we import pylab
from pylab import *

# we import seaborn
import seaborn as sns

def f(t, tau=4, sigma=1/2):
    return cos(t*sigma) * exp(-t*tau)

t = linspace(0, 4, num=1e3)

taus = [1, 2, 3, 4, 5]
y = [f(t, tau=tau) for tau in taus]

figure()
for i, tau in enumerate(taus):
    plot(t, y[i], label=r'$\tau = {0}$'.format(tau))

legend(loc='best')
show()

# we use: https://github.com/stsievert
# use: https://stsievert.com/blog/2015/09/01/matlab-to-python/






# use: wavSI1027
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSI1027', sr=None)

print(librosa.samples_to_time(len(y), sr))

D = librosa.stft(y)
#print(D.shape, D.dtype)

S, phase = librosa.magphase(D)
#print(S.dtype, phase.dtype, np.allclose(D, S * phase))

plt.figure(figsize=(14, 4))

plt.subplot(211)

#librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
#                                                 ref=np.max), y_axis='log', x_axis='time')

#librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
#                                                 ref=np.max), y_axis='linear', x_axis='time')

librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
                                                 ref=np.max), y_axis='linear')

plt.title('Spectrogram - SI1027')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
#plt.show()

# use: wavSI1657
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSI1657', sr=None)

print(librosa.samples_to_time(len(y), sr))

D = librosa.stft(y)
#print(D.shape, D.dtype)

S, phase = librosa.magphase(D)
#print(S.dtype, phase.dtype, np.allclose(D, S * phase))

plt.subplot(212)

#librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
#                                                 ref=np.max), y_axis='log', x_axis='time')

#librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
#                                                 ref=np.max), y_axis='linear', x_axis='time')

librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
                                                 ref=np.max), y_axis='linear')

plt.title('Spectrogram - SI1657')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()



# use: wavSI1657
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSI1657', sr=None)
print(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max).shape)

storeAll = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max)

# use: wavSI1027
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSI1027', sr=None)
print(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max).shape)

#storeAll = [storeAll, librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max)]
storeAll = np.concatenate((storeAll, librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max)), axis=1)

print('')
print(np.shape(storeAll))



# use: wavSI1657
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSI1657', sr=None)
#print(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max).shape)
storeAll = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max)

# use: wavSI1027
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSI1027', sr=None)
#print(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max).shape)
storeAll = np.concatenate((storeAll, librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max)), axis=1)

# use: wavSI648
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSI648', sr=None)
#print(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max).shape)
storeAll = np.concatenate((storeAll, librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max)), axis=1)

# use: wavSA2
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA2', sr=None)
#print(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max).shape)
storeAll = np.concatenate((storeAll, librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max)), axis=1)

# use: wavSA1
y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA1', sr=None)
#print(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max).shape)
storeAll = np.concatenate((storeAll, librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)), ref=np.max)), axis=1)

#print(np.shape(storeAll))

print('')
print(np.shape(storeAll))



# we use: pyaudio
# use: https://people.csail.mit.edu/hubert/pyaudio/

# we use: speech_recognition
# use: https://pypi.org/project/SpeechRecognition/2.1.3/

import speech_recognition as sr

from os import path

#AUDIO_FILE = "audio.wav"
AUDIO_FILE = "/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA1"

r = sr.Recognizer()

with sr.AudioFile(AUDIO_FILE) as source:
 audio = r.record(source)

try:
   print("Sphinx thinks you said " + r.recognize_sphinx(audio))

except sr.UnknownValueError:
   print("Sphinx could not understand audio")

except sr.RequestError as e:
  print("Sphinx error; {0}".format(e))



# use: https://github.com/Apress/Deep-Learning-Apps-Using-Python/blob/master/Chapter11_Speech%20to%20text%20and%20vice%20versa/Speech%20to%20Text%20API%20and%20Text%20to%20Speech.ipynb

import speech_recognition as sr

from os import path

#AUDIO_FILE = "audio.wav"
AUDIO_FILE = "/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA1"

# use the audio file as the audio source
r = sr.Recognizer()

with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source) # read the entire audio file

# recognize speech using Sphinx
try:
    print("Sphinx thinks you said " + r.recognize_sphinx(audio))

except sr.UnknownValueError:
    print("Sphinx could not understand audio")

except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))

# we now use: https://www.apress.com/gb/book/9781484235157

# use: https://github.com/Apress/Deep-Learning-Apps-Using-Python
# we use: https://github.com/Apress/Deep-Learning-Apps-Using-Python/blob/master/Chapter11_Speech%20to%20text%20and%20vice%20versa/Speech%20to%20Text%20API%20and%20Text%20to%20Speech.ipynb

# we use: pyaudio
import pyaudio
import wave

def record_audio(RECORD_SECONDS, WAVE_OUTPUT_FILENAME):
    # --------- SETTING PARAMS FOR OUR AUDIO FILE ------------#
    FORMAT = pyaudio.paInt16  # format of wave
    CHANNELS = 2  # no. of audio channels

    RATE = 44100  # frame rate
    CHUNK = 1024  # frames per audio sample
    # --------------------------------------------------------#

    # creating PyAudio object
    audio = pyaudio.PyAudio()

    # open a new stream for microphone
    # It creates a PortAudio Stream Wrapper class object
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    # ----------------- start of recording -------------------#
    print("Listening...")

    # list to save all audio frames
    frames = []

    for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
        # read audio stream from microphone
        data = stream.read(CHUNK)
        # append audio data to frames list
        frames.append(data)

    # ------------------ end of recording --------------------#
    print("Finished recording.")

    stream.stop_stream()  # stop the stream object
    stream.close()  # close the stream object
    audio.terminate()  # terminate PortAudio

    # ------------------ saving audio ------------------------#

    # create wave file object
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')

    # settings for wave file object
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))

    # closing the wave file object
    waveFile.close()

def read_audio(WAVE_FILENAME):
    # function to read audio(wav) file
    with open(WAVE_FILENAME, 'rb') as f:
        audio = f.read()
    return audio

try:
    print("Google Speech Recognition thinks you said " + r.recognize_google(audio))

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

#import requests
#import json

# Wit speech API endpoint
#API_ENDPOINT = 'https://api.wit.ai/speech'

# Wit.ai api access token
#wit_access_token = 'ERGSGTLIC3RLAUEFTULLKIXRUUA3GOXG'

#def RecognizeSpeech(AUDIO_FILENAME, num_seconds=5):
#    # record audio of specified length in specified audio file
#    record_audio(num_seconds, AUDIO_FILENAME)
#
#    # reading audio
#    audio = read_audio(AUDIO_FILENAME)
#
#    # defining headers for HTTP request
#    headers = {'authorization': 'Bearer ' + wit_access_token, 'Content-Type': 'audio/wav'}
#
#    # making an HTTP post request
#    resp = requests.post(API_ENDPOINT, headers=headers, data=audio)
#
#    # converting response content to JSON format
#    data = json.loads(resp.content)
#
#    # get text from data
#    text = data['_text']
#
#    # return the text
#    return text
#
#if __name__ == "__main__":
#    text = RecognizeSpeech('audio.wav', 4)
#    print("\nYou said: {}".format(text))



# we now use MFCCs, Mel-frequency cepstral coefficients

# we use the library package "python_speech_features"
from python_speech_features import mfcc

# delta is the difference
from python_speech_features import delta

# we use the log filterbank
from python_speech_features import logfbank

import scipy.io.wavfile as wav

(fs, signal) = wav.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA1')
#(fs, signal) = wav.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA2')

#mfccfeatures = mfcc(signal, fs)

# compute the MFCCs
mfccfeatures = mfcc(signal, fs)

# we compute the delta MFCCs
dmfccfeatures = delta(mfccfeatures, 2)
# researchers usually use MFCCs, Delta MFCCs and Delta Delta MFCCs

fbankfeature = logfbank(signal, fs)

print(fbankfeature)

import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

#fs, sig = wavfile.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA1')
#fs, sig = wavfile.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA2')

# we use wavSI1657 from TIMIT
fs, sig = wavfile.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSI1657')

freq, times, spectrogram = signal.spectrogram(sig, fs, window='hann', nfft=512)

plt.figure()
plt.pcolormesh(times, freq, np.log(spectrogram))

plt.ylabel('Frequency (kHz)')
plt.xlabel('Time (s)')
plt.show()



# we use keras for neural networks (NNs)
import keras
from keras.layers.core import Dense, Activation

from keras.models import Sequential
model = Sequential()

# the input layer is of dimension 4
# the hidden layer is of dimension 10
model.add(Dense(output_dim=10, input_dim=4))

model.add(Activation('relu'))

# the output layer is of dimension 3
model.add(Dense(output_dim=3))

# softmax output layer, softmax, normalized exponential
model.add(Activation('softmax'))

#model.summary

# cross-entropy (CE) cost function
# stochastic gradient descent (SGD)
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# we use .fit()
#model.fit(x_train, y_train, epochs=20, batch_size=128)
# train/fit the model






import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator

mean = [3, 4]
cov = [[1, 5], [5,  10]]

# PCA, eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(cov)

print(eigvals)
print(eigvecs)



X = np.random.multivariate_normal(mean, cov, 100).T

fig = plt.figure()
ax1 = fig.add_subplot(131)

ax1.scatter(X[0], X[1])
ax1.grid()

ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)

ax1.axvline(0, c='k')
ax1.axhline(0, c='k')

ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

ax1.set_title('Original data')

mean = np.mean(X)
print('mean', mean)
centred = X - mean

ax2 = fig.add_subplot(132)

ax2.scatter(centred[0], centred[1])
ax2.grid()

ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)

ax2.axvline(0, c='k')
ax2.axhline(0, c='k')

ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

ax2.set_title('Step 1: Centre data around mean')


ranges = np.ptp(X)
print('ranges', ranges)
scaled = np.divide(centred, ranges)

ax3 = fig.add_subplot(133)

ax3.scatter(scaled[0], scaled[1])
ax3.grid()

ax3.set_xlim(-1, 1)
ax3.set_ylim(-1, 1)

ax3.axvline(0, c='k')
ax3.axhline(0, c='k')

ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

ax3.set_title('Step 2: Scale data')

plt.show()

covscaled = np.matmul(scaled, scaled.T)

print('\n\nScaled feature covariance')
print(covscaled)

eigvals, eigvecs = np.linalg.eig(covscaled)

print('eigvals')
print(eigvals)

print('Eigvecs')
print(eigvecs)

fig2 = plt.figure()
ax4 = fig2.add_subplot(131)
ax4.grid()

ax4.scatter(scaled[0], scaled[1])

ax4.set_xlim(-1, 1)
ax4.set_ylim(-1, 1)

ax4.quiver(eigvals[0]*eigvecs[0,0], eigvals[0]*eigvecs[1, 0], scale_units='xy', scale=5)
ax4.quiver(eigvals[1]*eigvecs[0,1], eigvals[1]*eigvecs[1, 1], scale_units='xy', scale=5)

ax4.axvline(0, c='k')
ax4.axhline(0, c='k')

ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
ax4.yaxis.set_major_locator(MaxNLocator(integer=True))

ax4.set_title('Step 3: Determine eigenvectors of covariance matrix')

eigvecs = eigvecs[:, -1::-1]
transformed = np.matmul(eigvecs.T, scaled)

ax5 = fig2.add_subplot(132)
ax5.grid()

ax5.scatter(transformed[0], transformed[1])

ax5.set_xlim(-1, 1)
ax5.set_ylim(-1, 1)

ax5.axvline(0, c='k')
ax5.axhline(0, c='k')

ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
ax5.yaxis.set_major_locator(MaxNLocator(integer=True))

ax5.set_title('Step 4: Make eigenvectors the axes')

reduced = np.matmul(eigvecs[:, 0], scaled)

ax6 = fig2.add_subplot(133)
ax6.grid()

ax6.scatter(reduced, np.zeros_like(reduced))

ax6.set_xlim(-1, 1)
ax6.set_ylim(-1, 1)

ax6.axvline(0, c='k')
ax6.axhline(0, c='k')

ax6.xaxis.set_major_locator(MaxNLocator(integer=True))
ax6.yaxis.set_major_locator(MaxNLocator(integer=True))

ax6.set_title('Step 5: Throw away higher principal components')
plt.show()



# we define input
input_list = [0, 2, 4, 4, 1, 5, 2]

# initialize "model_memory"
model_memory = [0] * len(input_list)
# we initialize the memory

# we store
loc_write = 0
for value in input_list:
    model_memory[loc_write] = value
    loc_write += 1

# we write
loc_read = 0
while loc_read < loc_write:
    print(model_memory[loc_read])
    loc_read += 1



# we use CNTK
# use: cntk.sequence.slice

# sequence model: use every 3rd frame

n_channels = 12

input_var = cntk.sequence.input_variable([cntk.FreeDimension, n_channels])

model = cntk.slice(input_var, axis=0, begin_index=0, end_index=0, strides=3)

x = np.random.rand(1, 6, n_channels).astype(np.float32)

#print(model.eval({model.arguments[0]: x}))



# use: TensorFlow pre-trained model

# use pre-trained model from PyTorch
# we use: resnet_v1_50 on the ImageNet validation setl, pytorch pretrained resnet_50 is 76.15%

trainloader = imagenet_traindata(args.batch_size)
testloader = imagenet_testdata(args.batch_size)

MainModel = imp.load_source('MainModel', "tf_resnetv1_50_to_pth.py")

# load pre-trained model from PyTorch
model = torch.load('tf_resnetv1_50_to_pth.pth')

model = nn.DataParallel(model)
model = model.cuda()

print(model)

trainloader = imagenet_traindata(args.batch_size)
testloader = imagenet_testdata(args.batch_size)

MainModel = imp.load_source('MainModel', "tf_resnetv1_50_to_pth.py")
model = torch.load('tf_resnetv1_50_to_pth.pth')

model = nn.DataParallel(model)
model = model.cuda()

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    print('Test: *Loss {loss.avg:.4f} \tPrec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))

    return top1.avg

validate(testloader, model, criterion)



import tensorflow as tf
sess = tf.Session()

my_graph = tf.Graph()

with my_graph.as_default():
    variable = tf.Variable(30, name='navin')

    initialize = tf.global_variables_initializer()

with tf.Session(graph = my_graph) as sess:
    sess.run(initialize)

    print(sess.run(variable))



import numpy as np
import os

from keras.datasets import cifar10

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

# we use adaptive momentum, we use Adam
from keras.optimizers import adam

from keras.utils import np_utils

# we now load data
np.random.seed(100)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 3072)
x_train = (x_train - np.mean(x_train)) / np.std(x_train)

x_test = x_test.reshape(10000, 3072)
x_test = (x_test - np.mean(x_test)) / np.std(x_test)

# use one-hot encoding

labels = 10

# we use one-hot vectors
y_train = np_utils.to_categorical(y_train, labels)

# use one-hot vectors
y_test = np_utils.to_categorical(y_test, labels)

# we use classification, we use the cross-entropy (CE) cost function
# we use the CE cost function, we use one-hot encoding



model = Sequential()
model.add(Sense(512, input_shape=(3072,)))

# we use the ReLU activation function
model.add(Activation('relu'))

# use dropout to reduce overfitting
model.add(Dropout(0.4))
# we use dropout for better generalization

model.add(Dense(120))
model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(labels))

# the output layer is a sigmoid layer
model.add(Activation('sigmoid'))

adam = adam(0.1)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy: ', score[1])

model.predict_classes(x_test)

model.summary

#model.save('model.h5')
#jsonModel = model.to_json()

#model.save_weights('modelWeights.h5')

#modelWt = model.load_weights('modelWeight.h5')



# we use Kaggle, Kaggle competitions
# use: https://medium.com/@faizanahemad/participating-in-kaggle-data-science-competitions-part-1-step-by-step-guide-and-baseline-model-5b0c6973022a

# we use: https://www.kaggle.com/questions-and-answers/41211
# we also use: https://www.linkedin.com/pulse/machine-learning-whats-inside-box-randy-lao/?published=t

# article about Kaggle: https://medium.com/@faizanahemad/participating-in-kaggle-data-science-competitions-part-1-step-by-step-guide-and-baseline-model-5b0c6973022a

# use the "No free Hunch" Blog: http://blog.kaggle.com/2014/08/01/learning-from-the-best/
# we also use: http://blog.kaggle.com/2017/05/25/two-sigma-financial-modeling-challenge-winners-interview-2nd-place-nima-shahbazi-chahhou-mohamed/

# Kaggle, nd1511: https://www.kaggle.com/nd1511
# nd1511, Nikolaos Dionelis: https://www.kaggle.com/nd1511



# we read stored data

# we now read stored data in Python
# In order to read the data in a Python script, a function like the following can easily read the data.

def read_air_and_filters_xy(h5_files, framesize=None, get_pow_spec=True,
                            max_air_len=None, fs=None, forced_fs=None, keep_ids=None,
                            start_at_max=True, max_air_read=None):
    latest_file = '../results_dir/training_test_data.h5'
    from os.path import isfile

    import numpy as np

    from h5py import File

    # use: resampy
    from resampy import resample

    ids = None

    x = None

    all_boudnaries = None

    if forced_fs is None:
        forced_fs = fs

    resample_op = lambda x: x

    if not forced_fs == fs:
        resample_op = lambda x: np.array(resample(np.array(x.T, dtype=np.float64), fs, forced_fs, 0)).T

    if max_air_read is not None:
        if fs is None:
            raise AssertionError('Cannot work with max_air_read without fs')

            max_air_read_samples = int(np.ceil(fs * max_air_read))

    for i, this_h5 in enumerate(h5_files):
        print
        "Reading : " + this_h5 + " @ " + str(i + 1) + " of " + str(len(h5_files)),
        hf = File(this_h5, 'r')

        names = np.array(hf.get('names'))

        airs = np.array(hf.get('airs')).T

        boundaries = np.array(hf.get('boundary_ids')).T

        if i > 0:
            ids = np.concatenate((ids, names))
        else:
            ids = names

        print("Got " + str(airs.shape))

        airs = resample_op(airs)

        if max_air_read is not None:
            airs = airs[:, 0:max_air_read_samples]
        if i > 0:
            if x.shape[1] < airs.shape[1]:
                npads = -x.shape[1] + airs.shape[1]
                x = np.concatenate((x, np.zeros((x.shape[0], npads)).astype(x.dtype)), axis=1)
                x = np.concatenate((x, airs), axis=0)
            else:
                if x.shape[1] > airs.shape[1]:
                    npads = x.shape[1] - airs.shape[1]
                    airs = np.concatenate((airs, np.zeros((airs.shape[0], npads)).astype(
                        airs.dtype)), axis=1)
                x.resize((x.shape[0] + airs.shape[0], x.shape[1]))
                x[-airs.shape[0]:, :] = airs
        else:
            x = np.array(airs)

        if i > 0:
            all_boudnaries = np.concatenate((all_boudnaries, boundaries), axis=0)
        else:
            all_boudnaries = boundaries

    class_names = np.unique(all_boudnaries)

    y = np.zeros((all_boudnaries.shape[0], class_names.size)).astype(bool)

    for i, cname in enumerate(class_names):
        y[np.any(all_boudnaries == cname, axis=1), i] = True

    if keep_ids is not None:
        y = y[:, np.in1d(class_names.astype(int), np.array(keep_ids).astype(int))]

    if fs is not None:
        print('Got ' + str(x.shape[0]) + ' AIRs of duration ' + str(x.shape[1] / float(fs)))
    else:
        print('Got ' + str(x.shape[0]) + ' AIRs of length ' + str(x.shape[1]))

    x = data_post_proc(x, fs, start_at_max, framesize, get_pow_spec, max_air_len)

    print('Left with ' + str(x.shape) + ' AIRs data ')

    ids = ids.astype(str)

    class_names = class_names.astype(str)

    return (x, y), ids, class_names



# split the data for training and testing

# training and testing
# Once we have acquired a number of examples for our task, we can start training your network.

def get_split_data(air_files, train_ratio=.85, val_ratio=.075,
                   test_ratio=.075, stratified=True, framesize=None, print_set_report=True,
                   clustered_data=True, **kwargs):
    import numpy as np

    from myutils_reverb import read_li8_file

    val_sum = train_ratio + test_ratio + val_ratio

    if not isclose(val_sum, 1.):
        raise AssertionError('Ratios should sum to 1.00 and not ' + str(val_sum))

    (x, y), ids, class_names = read_air_and_filters_xy(air_files,
                                                       framesize=framesize, **kwargs)

    if stratified:
        from sklearn.model_selection import StratifiedShuffleSplit as splitter

        y_new = np.zeros((y.shape[0],)).astype('int64')

        uvals = np.unique(y, axis=0)

        for i in range(uvals.shape[0]):
            y_new[np.all(y == uvals[i, :], axis=1)] = i
    else:
        from sklearn.model_selection import ShuffleSplit as splitter
        y_new = y

    sss = splitter(n_splits=1, test_size=test_ratio, random_state=50)

    for train_val_index, test_index in sss.split(np.zeros_like(y_new), y_new):
        pass

    sss_val = splitter(n_splits=1, test_size=(val_ratio) / (val_ratio + train_ratio),
                       random_state=50)
    for train_index, val_index in sss_val.split(np.zeros_like(y_new[train_val_index]),
                                                y_new[train_val_index]):
        pass

    train_index = train_val_index[train_index]

    val_index = train_val_index[val_index]

    if print_set_report:
        print_split_report(y, idx_list=(train_index, val_index, test_index),
                           set_names=('Train', 'Val', 'Test'))

    return (x[train_index, :], y[train_index, :]), ids[train_index], \
           (x[test_index, :], y[test_index, :]), ids[test_index], \
           (x[val_index, :], y[val_index, :]), ids[val_index], \
           (x, y), ids, \
           class_names



# we now define our Keras model

# create a Keras model
def get_model(input_dims, n_outputs, dense_width=128):
    from keras.models import Sequential

    from keras.layers import Dense, InputLayer, Reshape, Dropout, TimeDistributed, GRU

    default_activation = 'relu'

    model = Sequential()

    model.add(InputLayer(input_shape=tuple(list(input_dims) + [1])))
    model.add(Reshape((model.output_shape[1], model.output_shape[2])))

    model.add(TimeDistributed(Dense(dense_width, activation=default_activation)))
    model.add(TimeDistributed(Dense(dense_width, activation=default_activation)))

    model.add(TimeDistributed(Dense(dense_width, activation=default_activation)))
    model.add(TimeDistributed(Dense(dense_width, activation=default_activation)))

    model.add(Dropout(0.05))
    model.add(TimeDistributed(Dense(dense_width, activation=default_activation)))

    model.add(Dropout(0.05))
    model.add(GRU(dense_width, activation=default_activation))

    model.add(Dense(n_outputs, activation='sigmoid'))

    return model



# we use an apropriate loss for the problem

# we use multi-label detection
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x, y, epochs=1000, validation_data=val_data,
          batch_size=batch_size_base, shuffle=True, sample_weight=sample_weights, verbose=2, callbacks=callbacks)



# we now use callbacks

# we use callbacks in Python
callbacks = []
callbacks.append(EarlyStopping(monitor='loss', min_delta=0, patience=loss_patience, verbose=0, mode='auto'))

callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=val_patience, verbose=0, mode='auto'))

tensordir = '../results/tensorlogs_dir'

# use TensorBoard
callbacks.append(TensorBoard(log_dir=tensordir, histogram_freq=0, batch_size=batch_size_base,
                write_graph=True, write_grads=False, write_images=False,
                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))



# evaluation
# we evaluate our DL model

def get_scores(y_pred, y_gt, beta=1):
    import numpy as np

    if y_pred.ndim > 1 or y_gt.ndim > 1:
        raise ValueError('Expected 1D inputs')

    if not y_pred.size == y_gt.size:
        raise ValueError('Expected 1D inputs of same size')

    tp = np.logical_and(y_pred, y_gt)

    # compute the precision
    precision = tp.sum() / y_pred.sum().astype(float)

    # compute the recall
    recall = tp.sum() / y_gt.sum().astype(float)

    fbeta = (1.0 + beta ** 2) * (precision * recall) / float((precision * beta ** 2) + recall)

    fp = np.sum(np.logical_and(y_pred, np.logical_not(y_gt)))
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y_gt))

    tp = np.sum(np.logical_and(y_pred, y_gt))
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_gt)))

    fpr = fp / float(fp + tn)
    fnr = fn / float(fn + tp)

    metrics = ('F' + str(beta), 'Precision', 'Recall', 'False Positive', 'False Negative',
               'False Positive Rate', 'False Negative Rate')

    metric_values = (fbeta, precision, recall, fp, fn, fpr, fnr)

    return metrics, metric_values

# we now use TensorBoard
# we use: https://www.tensorflow.org/guide/summaries_and_tensorboard

