
# we use the TIMIT speech database

# we use TIMIT
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
display.specshow(logPowerSpectrum, y_axis='linear', x_axis='time', sr=sr)

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

display.specshow(logPowerSpectrum, y_axis='linear', x_axis='time', sr=sr)

plt.title('Log-Spectrogram')

plt.show()



plt.figure(figsize=(14, 4))

#librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
#                                                 ref=np.max), y_axis='log', x_axis='time')

librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
                                                 ref=np.max), y_axis='linear', x_axis='time')

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

librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=256, window='hann', center=True)),
                                                 ref=np.max), y_axis='linear', x_axis='time')

plt.title('Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()

plt.show()


