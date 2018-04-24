
from sklearn import datasets

import numpy as np



# we use the iris dataset
iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target

#print('Class Labels', y)
print('Class Labels', np.unique(y))



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

import numpy

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

mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)









# we use speechpy
# we use: https://github.com/astorfi/speechpy

import scipy.io.wavfile as wav
import numpy as np

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



############# Extract MFCC features #############
mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)

print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)

print('mfcc feature cube shape=', mfcc_feature_cube.shape)



############# Extract logenergy features #############
logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)

print('logenergy features=', logenergy.shape)








# we use the Chime Challenge

# we use the Chime Challenge
# we can use data from the Chime Challenge

# in MATLAB, we have:
# [y, fs] = readwav('/Volumes/Maxtor/CHiME5/audio/train/S03_U01.CH1.wav');
# size(y), fs
# %soundsc(y, fs)
# %clear sound
# %figure; plot((1:length(y))*(1/fs), y); axisenlarge; figbolden; xlabel('Time (s)'); figbolden; ylabel('Amplitude'); figbolden;
# %figure; plot((1:length(y))*(1/fs)/60, y); axisenlarge; figbolden; xlabel('Time (m)'); figbolden; ylabel('Amplitude'); figbolden;
# figure; plot((1:length(y))*(1/fs)/(60*60), y); axisenlarge; figbolden; xlabel('Time (h)'); figbolden; ylabel('Amplitude'); figbolden;







