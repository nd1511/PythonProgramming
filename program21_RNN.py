
# we use: http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf
# we use: F. Chollet, Deep learning with Python

# RNN
# an RNN is a for loop that reuses quantities computed during the previous iteration of the loop

import numpy as np

#state_t = 0
#for input_t in input_sequence:
#output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
#state_t = output_t

timesteps = 100
# the input sequence has 100 timesteps

input_features = 32
# the dimensionality of the output is 32

output_features = 64
# the dimensionality of the output is 64



# we define the input
inputs = np.random.random((timesteps, input_features))
# the input is random noise

# initialize the state
state_t = np.zeros((output_features,))
# the state is initialized to zero

W = np.random.random((output_features, input_features))

U = np.random.random((output_features, output_features))

b = np.random.random((output_features,))

successive_outputs = []



for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)

    # we define the output
    #output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    # we use tanh

    successive_outputs.append(output_t)

    state_t = output_t

    final_output_sequence = np.concatenate(successive_outputs, axis=0)






from keras.layers import SimpleRNN
# we use SimpleRNN

# SimpleRNN has inputs: (batch_size, timesteps, input_features)



from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()

model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))

model.summary()



model = Sequential()

model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))

model.summary()



# we now stack several recurrent layers one after the other
model = Sequential()

model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))

model.summary()



# we use IMDB
from keras.datasets import imdb

# we use the IMDB movie review classification problem

from keras.preprocessing import sequence

max_features = 10000

maxlen = 500

batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)



# we use an Embedding layer and a SimpleRNN layer
# we use: SimpleRNN

from keras.layers import Dense

model = Sequential()

model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
# we train the model

# we see:
# 5120/20000 [======>.......................] - ETA: 17s - loss: 0.6916 - acc: 0.5273
# 5248/20000 [======>.......................] - ETA: 17s - loss: 0.6917 - acc: 0.5274
# 5376/20000 [=======>......................] - ETA: 17s - loss: 0.6916 - acc: 0.5283
# 5504/20000 [=======>......................] - ETA: 17s - loss: 0.6913 - acc: 0.5305
# 5632/20000 [=======>......................] - ETA: 17s - loss: 0.6912 - acc: 0.5314

# 13696/20000 [===================>..........] - ETA: 7s - loss: 0.2009 - acc: 0.9265
# 13824/20000 [===================>..........] - ETA: 7s - loss: 0.2010 - acc: 0.9266
# 13952/20000 [===================>..........] - ETA: 7s - loss: 0.2018 - acc: 0.9261

# Epoch 8/10



# we plot the training and validation loss and accuracy

# we use plt to plot figures
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

# we define the epochs
epochs = range(1, len(acc) + 1)

# we plot the training and validation accuracy

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')
plt.legend()

# we now plot the training and validation loss

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()

plt.show()




