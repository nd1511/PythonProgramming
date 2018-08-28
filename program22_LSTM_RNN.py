
# we use: F. Chollet's book "Deep learning with Python"

# we use: http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf
# we use: F. Chollet's book "Deep learning with Python"

# LSTM = Long Short Term memory, LSTM recurrent neural networks (RNNs)
# GRU = Gated Recurrent Unit, GRU RNNs
# we use: LSTM RNNs and GRU RNNs

# LSTM RNNs are used in (https://research.google.com/pubs/archive/44312.pdf)
# GRU RNNs are used in (https://pdfs.semanticscholar.org/702e/aa99bcb366d08d7f450ed7e354f9f6920b23.pdf)

# GRU RNNs along with mixture density networks (MDNs) are used in (https://pdfs.semanticscholar.org/702e/aa99bcb366d08d7f450ed7e354f9f6920b23.pdf)

# we use: LSTM RNNs and GRU RNNs
# LSTM RNNs and GRU RNNs are better than (vanilla) RNNs



# we use the kapre library
# we use: https://github.com/keunwoochoi/kapre

# kapre and Keras can be used together

# Deep neural networks (DNNs) are used in one of the papers in (https://drive.google.com/drive/folders/1GSMA7KPnJQ0LFBu3F3t7VmybFtAn580U)
# https://www.commsp.ee.ic.ac.uk/~sap/people-nikolaos-dionelis/



# we use the terminal and the command line to download a file

# we use the terminal
# cd ~/Downloads
# mkdir jena_climate
# cd jena_climate
# wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
# unzip jena_climate_2009_2016.csv.zip

import os

data_dir = '/users/dionelisnikolaos/Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

# ['"Date Time"', '"p (mbar)"', '"T (degC)"', '"Tpot (K)"', '"Tdew (degC)"', '"rh (%)"', '"VPmax (mbar)"', '"VPact (mbar)"', '"VPdef (mbar)"',
# '"sh (g/kg)"', '"H2OC (mmol/mol)"', '"rho (g/m**3)"', '"wv (m/s)"', '"max. wv (m/s)"', '"wd (deg)"']
# 420551

import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))

# we now parse the data
# we convert the 420551 lines of data into a Numpy array

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values



from matplotlib import pyplot as plt

temp = float_data[:, 1]

plt.plot(range(len(temp)), temp)
plt.show()

plt.plot(range(1440), temp[:1440])
plt.show()



# we normalize the data

mean = float_data[:200000].mean(axis=0)
float_data -= mean

std = float_data[:200000].std(axis=0)
float_data /= std



def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1

    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)

        else:
            if i + batch_size >= max_index:
                i = min_index + lookback

            rows = np.arange(i, min(i + batch_size, max_index))

            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))

        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)

            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets



# training, validation and testing
# prepare the data

lookback = 1440
step = 6

delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)

val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step, batch_size=batch_size)

test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None, step=step, batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)



# baseline
# create a baseline algorithm

def evaluate_naive_method():
    batch_maes = []

    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]

        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)

    print(np.mean(batch_maes))

evaluate_naive_method()

celsius_mae = 0.29 * std[1]



from keras.models import Sequential
from keras import layers

from keras.optimizers import RMSprop

model = Sequential()

model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)



# plot the loss curves for validation and training

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()

plt.show()



from keras.models import Sequential
from keras import layers

from keras.optimizers import RMSprop

model = Sequential()

model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)



from keras.models import Sequential
from keras import layers

from keras.optimizers import RMSprop

model = Sequential()

# we create a GRU RNN
# GRU = Gated Recurrent Unit, GRU RNNs

# we use GRU
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1]))) model.add(layers.Dense(1))

# GRU RNNs are used in the Apple paper (https://pdfs.semanticscholar.org/702e/aa99bcb366d08d7f450ed7e354f9f6920b23.pdf)
# GRU RNNs along with mixture density networks (MDNs) are used in (https://pdfs.semanticscholar.org/702e/aa99bcb366d08d7f450ed7e354f9f6920b23.pdf)

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)



from keras.models import Sequential
from keras import layers

from keras.optimizers import RMSprop

model = Sequential()

model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=(None, float_data.shape[-1])))

model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)






# use bidirectional RNNs

# use a different model
# we now use bidirectional RNNs

# we use imdb
from keras.datasets import imdb

from keras.preprocessing import sequence
from keras import layers

from keras.models import Sequential

max_features = 10000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()

model.add(layers.Embedding(max_features, 128))

model.add(layers.LSTM(32))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# we fit the model
# we train the bidirectional RNN

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)



# training and evaluation

model = Sequential()

model.add(layers.Embedding(max_features, 32))

# we use Bidirectional
model.add(layers.Bidirectional(layers.LSTM(32)))
# we also use LSTM

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# we now train the RNN model
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)



# we use a GRU
# we create a bidirectional GRU RNN

from keras.models import Sequential
from keras import layers

from keras.optimizers import RMSprop

model = Sequential()

model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)






