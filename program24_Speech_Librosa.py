
# we use the TIMIT speech database

# we use the TIMIT database
# we use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2

# use: https://github.com/librosa/tutorial/blob/master/Librosa%20tutorial.ipynb

# we use: https://github.com/keunwoochoi/kapre/blob/master/examples/example_codes.ipynb
# we also use: https://github.com/keunwoochoi/kapre/blob/master/examples/prepare%20audio.ipynb

import numpy as np

#import keras
#import kapre

import librosa
from librosa import display

# we use plt to plot figures
import matplotlib.pyplot as plt

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






# we use: https://github.com/Imperial-College-Data-Science-Society/Neural-Networks/blob/master/slides/L2.Neural-Networks.pdf

# we use: https://github.com/Imperial-College-Data-Science-Society/Neural-Networks

# for MATLAB: https://github.com/dustinstansbury/medal
# we use: https://github.com/PhDP/mlbop/tree/master/MATLAB-18



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

