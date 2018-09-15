
# we use the TIMIT speech database

# we use the TIMIT database
# we use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2

# we use: https://github.com/librosa/tutorial/blob/master/Librosa%20tutorial.ipynb

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


