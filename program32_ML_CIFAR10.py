from __future__ import print_function
from __future__ import absolute_import

import os
import tempfile

import subprocess
import matplotlib.pyplot as plt

import numpy as np
import sonnet as snt

import tarfile
import tensorflow as tf

from six.moves import cPickle
from six.moves import urllib
from six.moves import xrange

# CIFAR-10 Dataset
data_path = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

local_data_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(local_data_dir)

url = urllib.request.urlopen(data_path)
archive = tarfile.open(fileobj=url, mode='r|gz')

archive.extractall(local_data_dir)
url.close()

archive.close()
print('extracted data files to %s' % local_data_dir)

# https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
# we use: https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb

def unpickle(filename):
    with open(filename, 'rb') as fo:
        return cPickle.load(fo, encoding='latin1')

def reshape_flattened_image_batch(flat_image_batch):
    return flat_image_batch.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])  # convert from NCHW to NHWC

def combine_batches(batch_list):
    images = np.vstack([reshape_flattened_image_batch(batch['data'])
                        for batch in batch_list])
    labels = np.vstack([np.array(batch['labels']) for batch in batch_list]).reshape(-1, 1)
    return {'images': images, 'labels': labels}

train_data_dict = combine_batches([
    unpickle(os.path.join(local_data_dir,
                          'cifar-10-batches-py/data_batch_%d' % i))
    for i in range(1, 5)])

valid_data_dict = combine_batches([
    unpickle(os.path.join(local_data_dir,
                          'cifar-10-batches-py/data_batch_5'))])

test_data_dict = combine_batches([
    unpickle(os.path.join(local_data_dir, 'cifar-10-batches-py/test_batch'))])

def cast_and_normalise_images(data_dict):
  """Convert images to floating point with the range [0.5, 0.5]"""
  images = data_dict['images']

  data_dict['images'] = (tf.cast(images, tf.float32) / 255.0) - 0.5
  return data_dict

data_variance = np.var(train_data_dict['images'] / 255.0)

def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
    for i in range(num_residual_layers):
        h_i = tf.nn.relu(h)

        h_i = snt.Conv2D(
            output_channels=num_residual_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="res3x3_%d" % i)(h_i)
        h_i = tf.nn.relu(h_i)

        h_i = snt.Conv2D(
            output_channels=num_hiddens,
            kernel_shape=(1, 1),
            stride=(1, 1),
            name="res1x1_%d" % i)(h_i)
        h += h_i

    return tf.nn.relu(h)

class Encoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name='encoder'):
        super(Encoder, self).__init__(name=name)

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens / 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")(x)
        h = tf.nn.relu(h)

        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")(h)
        h = tf.nn.relu(h)

        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_3")(h)

        h = residual_stack(
            h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        return h

class Decoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name='decoder'):
        super(Decoder, self).__init__(name=name)

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1")(x)

        h = residual_stack(
            h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

        h = snt.Conv2DTranspose(
            output_channels=int(self._num_hiddens / 2),
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")(h)
        h = tf.nn.relu(h)

        x_recon = snt.Conv2DTranspose(
            output_channels=3,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3")(h)

        return x_recon

tf.reset_default_graph()

# VQ-VAE, DeepMind, Google AI
# https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb

# Set hyper-parameters
batch_size = 32
image_size = 32

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU
#num_training_updates = 50000

#num_training_updates = 50000
num_training_updates = 100

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

# These hyper-parameters define the size of the model (number of parameters and layers).
# The hyper-parameters in the paper were (For ImageNet):
# batch_size = 128
# image_size = 128
# num_hiddens = 128
# num_residual_hiddens = 32
# num_residual_layers = 2

# This value is not that important, usually 64 works.
# This will not change the capacity in the information-bottleneck.
embedding_dim = 64

# The higher this value, the higher the capacity in the information bottleneck.
num_embeddings = 512

# commitment_cost should be set appropriately. It's often useful to try a couple
# of values. It mostly depends on the scale of the reconstruction cost
# (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.
commitment_cost = 0.25

# Use EMA updates for the codebook (instead of the Adam optimizer).
# This typically converges faster, and makes the model less dependent on choice
# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
vq_use_ema = False

decay = 0.99
learning_rate = 3e-4

# Data Loading
train_dataset_iterator = (
    tf.data.Dataset.from_tensor_slices(train_data_dict)
        .map(cast_and_normalise_images)
        .shuffle(10000)
        .repeat(-1)  # repeat indefinitely
        .batch(batch_size)).make_one_shot_iterator()

valid_dataset_iterator = (
    tf.data.Dataset.from_tensor_slices(valid_data_dict)
        .map(cast_and_normalise_images)
        .repeat(1)  # 1 epoch
        .batch(batch_size)).make_initializable_iterator()

train_dataset_batch = train_dataset_iterator.get_next()
valid_dataset_batch = valid_dataset_iterator.get_next()

def get_images(sess, subset='train'):
    if subset == 'train':
        return sess.run(train_dataset_batch)['images']
    elif subset == 'valid':
        return sess.run(valid_dataset_batch)['images']

# Build the modules
encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)

pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
                          kernel_shape=(1, 1),
                          stride=(1, 1),
                          name="to_vq")

if vq_use_ema:
    vq_vae = snt.nets.VectorQuantizerEMA(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay)
else:
    vq_vae = snt.nets.VectorQuantizer(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost)

# Process inputs with conv stack, finishing with 1x1 to get to correct size.
x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
z = pre_vq_conv1(encoder(x))

# vq_output_train["quantize"] are the quantized outputs of the encoder.
# That is also what is used during training with the straight-through estimator.
# To get the one-hot coded assignments use vq_output_train["encodings"] instead.

# These encodings will not pass gradients into to encoder,
# but can be used to train a PixelCNN on top afterwards.

# For training
vq_output_train = vq_vae(z, is_training=True)
x_recon = decoder(vq_output_train["quantize"])

recon_error = tf.reduce_mean((x_recon - x) ** 2) / data_variance  # Normalized MSE
loss = recon_error + vq_output_train["loss"]

# For evaluation, make sure is_training=False!
vq_output_eval = vq_vae(z, is_training=False)
x_recon_eval = decoder(vq_output_eval["quantize"])

# The following is a useful value to track during training.
# It indicates how many codes are 'active' on average.
perplexity = vq_output_train["perplexity"]

# Create optimizer and TF session.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss)
sess = tf.train.SingularMonitoredSession()

# Train
train_res_recon_error = []
train_res_perplexity = []

for i in xrange(num_training_updates):
    feed_dict = {x: get_images(sess)}

    results = sess.run([train_op, recon_error, perplexity],
                       feed_dict=feed_dict)

    train_res_recon_error.append(results[1])
    train_res_perplexity.append(results[2])

    if (i + 1) % 100 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))

        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()

f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_res_recon_error)

ax.set_yscale('log')
ax.set_title('NMSE.')

ax = f.add_subplot(1,2,2)
ax.plot(train_res_perplexity)
ax.set_title('Average codebook usage (perplexity).')

# Reconstructions
sess.run(valid_dataset_iterator.initializer)

train_originals = get_images(sess, subset='train')
train_reconstructions = sess.run(x_recon_eval, feed_dict={x: train_originals})

valid_originals = get_images(sess, subset='valid')
valid_reconstructions = sess.run(x_recon_eval, feed_dict={x: valid_originals})

def convert_batch_to_image_grid(image_batch):
  reshaped = (image_batch.reshape(4, 8, 32, 32, 3)
              .transpose(0, 2, 1, 3, 4)
              .reshape(4 * 32, 8 * 32, 3))
  return reshaped + 0.5

f = plt.figure(figsize=(16,8))
ax = f.add_subplot(2,2,1)

ax.imshow(convert_batch_to_image_grid(train_originals), interpolation='nearest')

ax.set_title('training data originals')
plt.axis('off')

ax = f.add_subplot(2,2,2)
ax.imshow(convert_batch_to_image_grid(train_reconstructions), interpolation='nearest')

ax.set_title('training data reconstructions')
plt.axis('off')

ax = f.add_subplot(2,2,3)
ax.imshow(convert_batch_to_image_grid(valid_originals), interpolation='nearest')

ax.set_title('validation data originals')
plt.axis('off')

ax = f.add_subplot(2,2,4)
ax.imshow(convert_batch_to_image_grid(valid_reconstructions), interpolation='nearest')

ax.set_title('validation data reconstructions')
plt.axis('off')
plt.pause(3)



import torch
import torch.nn as nn

input_dim = 5
hidden_dim = 10

n_layers = 1
lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# we use: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/

batch_size = 1
seq_len = 1

inp = torch.randn(batch_size, seq_len, input_dim)
hidden_state = torch.randn(n_layers, batch_size, hidden_dim)

cell_state = torch.randn(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)

out, hidden = lstm_layer(inp, hidden)

print("Output shape: ", out.shape)
print("Hidden: ", hidden)

seq_len = 3
inp = torch.randn(batch_size, seq_len, input_dim)

out, hidden = lstm_layer(inp, hidden)
print(out.shape)

# Obtaining the last output
out = out.squeeze()[-1, :]
print(out.shape)

# we now use: https://github.com/gabrielloye/LSTM_Sentiment-Analysis
# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/

#import bz2
#from collections import Counter

#import re
#import nltk

#import numpy as np
#nltk.download('punkt')

#train_file = bz2.BZ2File('../input/amazon_reviews/train.ft.txt.bz2')
#test_file = bz2.BZ2File('../input/amazon_reviews/test.ft.txt.bz2')

#train_file = train_file.readlines()
#test_file = test_file.readlines()



import bz2
from collections import Counter

import re
import nltk

import numpy as np
nltk.download('punkt')

train_file = bz2.BZ2File('../input/amazon_reviews/train.ft.txt.bz2')
test_file = bz2.BZ2File('../input/amazon_reviews/test.ft.txt.bz2')

train_file = train_file.readlines()
test_file = test_file.readlines()

num_train = 800000  # We're training on the first 800,000 reviews in the dataset
num_test = 200000  # Using 200,000 reviews from test set

train_file = [x.decode('utf-8') for x in train_file[:num_train]]
test_file = [x.decode('utf-8') for x in test_file[:num_test]]

# Extracting labels from sentences
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]

test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

# Some simple cleaning of data
for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d', '0', train_sentences[i])

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d', '0', test_sentences[i])

# Modify URLs to <url>
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in \
            train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in \
            test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

words = Counter()  # Dictionary that will map a word to the number of times it appeared in all the training sentences

for i, sentence in enumerate(train_sentences):
    train_sentences[i] = []

    for word in nltk.word_tokenize(sentence):  # Tokenize the words
        words.update([word.lower()])  # Convert all the words to lowercase
        train_sentences[i].append(word)

    if i%20000 == 0:
        print(str((i*100)/num_train) + "% done")

print("OK. Done.")

# Removing the words that only appear once
words = {k:v for k,v in words.items() if v>1}

# Sorting the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)

# Adding padding and unknown to our vocabulary so that they will be assigned an index
words = ['_PAD','_UNK'] + words

# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

for i, sentence in enumerate(train_sentences):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

for i, sentence in enumerate(test_sentences):
    # For test sentences, we have to tokenize the sentences as well
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)

    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]

    return features

seq_len = 200  # The length that the sentences will be padded/shortened to

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)

# Converting our labels into numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

split_frac = 0.5 # 50% validation, 50% test
split_id = int(split_frac * len(test_sentences))

val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

batch_size = 400

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
#is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
#if is_cuda:
#    device = torch.device("cuda")
#else:
#    device = torch.device("cpu")

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

vocab_size = len(word2idx) + 1
output_size = 1

embedding_dim = 400
hidden_dim = 512
n_layers = 2

model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.to(device)

lr=0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2
counter = 0
print_every = 1000
clip = 5
valid_loss_min = np.Inf

model.train()
for i in range(epochs):
    h = model.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1

        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)

        model.zero_grad()
        output, h = model(inputs, h)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []

            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])

                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)

                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(i + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

# Loading the best model
model.load_state_dict(torch.load('./state_dict.pt'))

num_correct = 0
test_losses = []
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])

    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)

    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    pred = torch.round(output.squeeze())  # Rounds the output to 0/1

    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())

    num_correct += np.sum(correct)

# use: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
print("Test loss: {:.3f}".format(np.mean(test_losses)))

test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))



import tensorflow
import matplotlib.pyplot as plt

import time
import numpy as np
import tensorflow as tf

from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data

# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

# https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959
# use: https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959

# MNIST: Four files are available on this site:
# train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
# train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
# t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
# t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

# From Terminal:
# cd mnist/
# gzip train-images-idx3-ubyte.gz -d
# gzip train-labels-idx1-ubyte.gz -d
# gzip t10k-images-idx3-ubyte.gz -d
# gzip t10k-labels-idx1-ubyte.gz -d

import os
import struct

def load_mnist(path2, kind='train'):
    labels_path = os.path.join(path2)
    images_path = os.path.join(path2)

# loading the data
#X_train, y_train = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/', kind='train')

#y_train = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/train-labels-idx1-ubyte')
#X_train = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/train-images-idx3-ubyte')
#print('Rows: {X_train.shape[0]},  Columns: {X_train.shape[1]}')

from mlxtend.data import loadlocal_mnist

X_train, y_train = loadlocal_mnist(
        images_path='/Users/dionelisnikolaos/Downloads/mnist/train-images-idx3-ubyte',
        labels_path='/Users/dionelisnikolaos/Downloads/mnist/train-labels-idx1-ubyte')

# loading the data
#X_test, y_test = load_mnist('./mnist/', kind='t10k')

#y_test = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/t10k-labels-idx1-ubyte')
#X_test = load_mnist('/Users/dionelisnikolaos/Downloads/mnist/t10k-images-idx3-ubyte')
#print('Rows: {X_test.shape[0]},  Columns: {X_test.shape[1]}')

X_test, y_test = loadlocal_mnist(
        images_path='/Users/dionelisnikolaos/Downloads/mnist/t10k-images-idx3-ubyte',
        labels_path='/Users/dionelisnikolaos/Downloads/mnist/t10k-labels-idx1-ubyte')

# mean centering and normalization:
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

#print(X_train.shape)
#print(y_train.shape)

#print(X_test.shape)
#print(y_test.shape)

#print(mean_vals.shape)
#print(std_val.shape)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10):
    img = X_train_centered[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_yticks([])
ax[0].set_xticks([])

plt.tight_layout()
plt.show()

np.random.seed(123)
tf.set_random_seed(123)

import tensorflow.contrib.keras as keras
y_train_onehot = keras.utils.to_categorical(y_train)

print('First 3 labels: ', y_train[:3])
print('\nFirst 3 labels (one-hot):\n', y_train_onehot[:3])

# https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959
y_train_onehot = keras.utils.to_categorical(y_train)

print('First 3 labels: ', y_train[:3])
print('\nFirst 3 labels (one-hot):\n', y_train_onehot[:3])

# initialize model
model = keras.models.Sequential()

# add input layer
model.add(keras.layers.Dense(
    units=50,
    input_dim=X_train_centered.shape[1],
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    activation='tanh'))

model.add(keras.layers.Dense(10, activation='softmax'))

# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# train model
history = model.fit(
    X_train_centered, y_train_onehot,
    batch_size=64, epochs=50,
    verbose=1, validation_split=0.1)

y_train_pred = model.predict_classes(X_train_centered, verbose=0)
print('First 3 predictions: ', y_train_pred[:3])

# calculate training accuracy
y_train_pred = model.predict_classes(X_train_centered, verbose=0)

correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]

#print('Training accuracy: {(train_acc * 100):.2f}')
print(train_acc)

# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# use: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

# calculate testing accuracy
y_test_pred = model.predict_classes(X_test_centered, verbose=0)

correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]

print(test_acc)

# 48192/54000 [=========================>....] - ETA: 0s - loss: 0.0803 - acc: 0.9801
# 49984/54000 [==========================>...] - ETA: 0s - loss: 0.0800 - acc: 0.9802
# 51904/54000 [===========================>..] - ETA: 0s - loss: 0.0798 - acc: 0.9802
# 53952/54000 [============================>.] - ETA: 0s - loss: 0.0794 - acc: 0.9802
# 54000/54000 [==============================] - 2s 28us/sample - loss: 0.0794 - acc: 0.9803 - val_loss: 0.1108 - val_acc: 0.9668

# First 3 predictions:  [5 0 4]
# 0.9799666666666667
# 0.9621



ds = tf.contrib.distributions

def sample_mog(batch_size, n_mixture=8, std=0.01, radius=1.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture)

    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))

    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)

    return data.sample(batch_size)

#sample_mog(128)
#print(sample_mog(128))

samplePoints = sample_mog(1000)
#print(samplePoints)

import matplotlib.pyplot as plt

#plt.plot([1,2,3,4])
#plt.plot(samplePoints[:,0], samplePoints[:,1])

tf.InteractiveSession()
samplePoints2 = samplePoints.eval()

plt.plot(samplePoints2[:,0], samplePoints2[:,1])
plt.xlabel('x')
plt.ylabel('y')

plt.show()
#plt.pause(2)

# np.exp(a)/np.sum(np.exp(a))
# use: np.exp(a)/np.sum(np.exp(a))

# https://github.com/samet-akcay/ganomaly
# we use: https://github.com/samet-akcay/ganomaly

# GANs - TRAIN GANOMALY
# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.875 AUC: 0.533 max AUC: 0.559
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.830 AUC: 0.531 max AUC: 0.559
# >> Training model Ganomaly.[Done]

# Namespace(anomaly_class='bird', batchsize=64, beta1=0.5, dataroot='', dataset='cifar10',
# device='gpu', display=False, display_id=0, display_port=8097, display_server='http://localhost',
# droplast=True, extralayers=0, gpu_ids=[0], isTrain=True, isize=32, iter=0, load_weights=False, lr=0.0002,
# manualseed=-1, metric='roc', model='ganomaly', name='ganomaly/cifar10', nc=3, ndf=64, ngf=64, ngpu=1, niter=15,
# nz=100, outf='./output', phase='train', print_freq=100, proportion=0.1, resume='', save_image_freq=100,
# save_test_images=False, w_bce=1, w_enc=1, w_rec=50, workers=8)

# Files already downloaded and verified
# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 4.057 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 4.791 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 3/15
#    Avg Run Time (ms/batch): 4.897 AUC: 0.519 max AUC: 0.519
# >> Training model Ganomaly. Epoch 4/15
#    Avg Run Time (ms/batch): 4.792 AUC: 0.502 max AUC: 0.519
# >> Training model Ganomaly. Epoch 5/15
#    Avg Run Time (ms/batch): 4.937 AUC: 0.536 max AUC: 0.536
# >> Training model Ganomaly. Epoch 6/15
#    Avg Run Time (ms/batch): 4.883 AUC: 0.498 max AUC: 0.536
# >> Training model Ganomaly. Epoch 7/15
#    Avg Run Time (ms/batch): 4.960 AUC: 0.503 max AUC: 0.536
# >> Training model Ganomaly. Epoch 8/15
#    Avg Run Time (ms/batch): 4.916 AUC: 0.559 max AUC: 0.559
# >> Training model Ganomaly. Epoch 9/15
#    Avg Run Time (ms/batch): 4.870 AUC: 0.522 max AUC: 0.559
# >> Training model Ganomaly. Epoch 10/15
#    Avg Run Time (ms/batch): 4.898 AUC: 0.539 max AUC: 0.559
#  65% 455/703 [00:16<00:08, 28.19it/s]Reloading d net
# >> Training model Ganomaly. Epoch 11/15
#    Avg Run Time (ms/batch): 4.900 AUC: 0.529 max AUC: 0.559
# >> Training model Ganomaly. Epoch 12/15
#    Avg Run Time (ms/batch): 4.856 AUC: 0.541 max AUC: 0.559
# >> Training model Ganomaly. Epoch 13/15
#    Avg Run Time (ms/batch): 4.910 AUC: 0.528 max AUC: 0.559
# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.875 AUC: 0.533 max AUC: 0.559
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.830 AUC: 0.531 max AUC: 0.559
# >> Training model Ganomaly.[Done]



# https://github.com/samet-akcay/ganomaly
# we use: https://github.com/samet-akcay/ganomaly

# Files already downloaded and verified
# >> Training model Ganomaly.
#    Avg Run Time (ms/batch): 274.149 AUC: 0.621 max AUC: 0.621
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 284.825 AUC: 0.649 max AUC: 0.649

# Namespace(anomaly_class='bird', batchsize=64, beta1=0.5, dataroot='', dataset='cifar10', device='gpu',
# display=False, display_id=0, display_port=8097, display_server='http://localhost', droplast=True, extralayers=0,
# gpu_ids=[0], isTrain=True, isize=32, iter=0, load_weights=False, lr=0.0002, manualseed=-1, metric='roc',
# model='ganomaly', name='ganomaly/cifar10', nc=3, ndf=64, ngf=64, ngpu=1, niter=15, nz=100, outf='./output',
# phase='train', print_freq=100, proportion=0.1, resume='', save_image_freq=100, save_test_images=False, w_bce=1,
# w_enc=1, w_rec=50, workers=8)

# Files already downloaded and verified
# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 4.100 AUC: 0.504 max AUC: 0.504
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 4.894 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 3/15
#    Avg Run Time (ms/batch): 4.904 AUC: 0.491 max AUC: 0.513
# >> Training model Ganomaly. Epoch 4/15
#    Avg Run Time (ms/batch): 4.850 AUC: 0.538 max AUC: 0.538
# >> Training model Ganomaly. Epoch 5/15
#    Avg Run Time (ms/batch): 4.849 AUC: 0.498 max AUC: 0.538
# >> Training model Ganomaly. Epoch 6/15
#    Avg Run Time (ms/batch): 4.865 AUC: 0.498 max AUC: 0.538
# >> Training model Ganomaly. Epoch 7/15
#    Avg Run Time (ms/batch): 4.863 AUC: 0.529 max AUC: 0.538
# >> Training model Ganomaly. Epoch 8/15
#    Avg Run Time (ms/batch): 4.862 AUC: 0.520 max AUC: 0.538
# >> Training model Ganomaly. Epoch 9/15
#    Avg Run Time (ms/batch): 4.898 AUC: 0.496 max AUC: 0.538
# >> Training model Ganomaly. Epoch 10/15
#    Avg Run Time (ms/batch): 4.885 AUC: 0.523 max AUC: 0.538
# >> Training model Ganomaly. Epoch 11/15
#    Avg Run Time (ms/batch): 4.917 AUC: 0.539 max AUC: 0.539
#   7% 48/703 [00:02<00:25, 26.05it/s]Reloading d net
# >> Training model Ganomaly. Epoch 12/15
#    Avg Run Time (ms/batch): 4.922 AUC: 0.547 max AUC: 0.547
# >> Training model Ganomaly. Epoch 13/15
#    Avg Run Time (ms/batch): 4.824 AUC: 0.516 max AUC: 0.547
# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.866 AUC: 0.542 max AUC: 0.547
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.872 AUC: 0.513 max AUC: 0.547
# >> Training model Ganomaly.[Done]

import matplotlib
from matplotlib import pyplot

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/Users/dionelisnikolaos/Downloads/GANomaly_image.png')
imgplot = plt.imshow(img)
#plt.pause(2)

img2 = mpimg.imread('/Users/dionelisnikolaos/Downloads/GANomaly_image2.png')
imgplot2 = plt.imshow(img2)
#plt.pause(2)

# Files already downloaded and verified
# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 274.149 AUC: 0.621 max AUC: 0.621
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 284.825 AUC: 0.649 max AUC: 0.649

# https://github.com/samet-akcay/ganomaly
# we use: https://github.com/samet-akcay/ganomaly

# Files already downloaded and verified
# >> Training model Ganomaly.
# 100% 703/703 [20:16<00:00,  1.73s/it]
# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 274.149 AUC: 0.621 max AUC: 0.621
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 284.825 AUC: 0.649 max AUC: 0.649
#   2% 11/703 [00:18<20:00,  1.73s/it]Process Process-34:

# CIFAR-10 dataset
from keras.datasets import cifar10

# load the CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

# https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3
# use: https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3

# Files already downloaded and verified
# >> Training model Ganomaly. Epoch 1/15
#   0% 1/703 [00:01<19:09,  1.64s/it]   Avg Run Time (ms/batch): 238.385 AUC: 0.587 max AUC: 0.587
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 455.465 AUC: 0.589 max AUC: 0.589
# 100% 702/703 [22:10<00:01,  1.74s/it]
# >> Training model Ganomaly. Epoch 3/15
#    Avg Run Time (ms/batch): 247.068 AUC: 0.647 max AUC: 0.647
# >> Training model Ganomaly. Epoch 4/15
#   0% 3/703 [00:07<30:09,  2.59s/it]   Avg Run Time (ms/batch): 254.772 AUC: 0.596 max AUC: 0.647
#  70% 494/703 [19:20<06:27,  1.86s/it]

# we use keras
from keras.datasets import mnist

# load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# use tensorflow
import tensorflow as tf

# MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8

import matplotlib.pyplot as plt
plt.imshow(x_train[image_index], cmap='Greys')

plt.pause(2)

#x_train.shape
print(x_train.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#input_shape = (64, 64, 1)
input_shape = (28, 28, 1)

# Ensure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)

print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])



# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))

model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

# use Adam, use adaptive momentum
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit the model
model.fit(x=x_train,y=y_train, epochs=10)

model.evaluate(x_test, y_test)

image_index = 4444

plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.pause(2)

pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())

# 12768/60000 [=====>........................] - ETA: 1:13 - loss: 0.0376 - acc: 0.9880
# 12832/60000 [=====>........................] - ETA: 1:12 - loss: 0.0376 - acc: 0.9880
# 12896/60000 [=====>........................] - ETA: 1:12 - loss: 0.0375 - acc: 0.9881
# 12960/60000 [=====>........................] - ETA: 1:12 - loss: 0.0373 - acc: 0.9881

# >> Training model Ganomaly. Epoch 13/15
#    Avg Run Time (ms/batch): 4.910 AUC: 0.528 max AUC: 0.559
# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.875 AUC: 0.533 max AUC: 0.559
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.830 AUC: 0.531 max AUC: 0.559
# >> Training model Ganomaly.[Done]

# Namespace(anomaly_class='bird', batchsize=64, beta1=0.5, dataroot='', dataset='cifar10',
# device='gpu', display=False, display_id=0, display_port=8097, display_server='http://localhost',
# droplast=True, extralayers=0, gpu_ids=[0], isTrain=True, isize=32, iter=0, load_weights=False, lr=0.0002,
# manualseed=-1, metric='roc', model='ganomaly', name='ganomaly/cifar10', nc=3, ndf=64, ngf=64, ngpu=1, niter=15,
# nz=100, outf='./output', phase='train', print_freq=100, proportion=0.1, resume='', save_image_freq=100,
# save_test_images=False, w_bce=1, w_enc=1, w_rec=50, workers=8)

# Files already downloaded and verified
# >> Training model Ganomaly. Epoch 1/15
#    Avg Run Time (ms/batch): 4.057 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 2/15
#    Avg Run Time (ms/batch): 4.791 AUC: 0.513 max AUC: 0.513
# >> Training model Ganomaly. Epoch 3/15
#    Avg Run Time (ms/batch): 4.897 AUC: 0.519 max AUC: 0.519
# >> Training model Ganomaly. Epoch 4/15
#    Avg Run Time (ms/batch): 4.792 AUC: 0.502 max AUC: 0.519
# >> Training model Ganomaly. Epoch 5/15
#    Avg Run Time (ms/batch): 4.937 AUC: 0.536 max AUC: 0.536
# >> Training model Ganomaly. Epoch 6/15
#    Avg Run Time (ms/batch): 4.883 AUC: 0.498 max AUC: 0.536
# >> Training model Ganomaly. Epoch 7/15
#    Avg Run Time (ms/batch): 4.960 AUC: 0.503 max AUC: 0.536
# >> Training model Ganomaly. Epoch 8/15
#    Avg Run Time (ms/batch): 4.916 AUC: 0.559 max AUC: 0.559
# >> Training model Ganomaly. Epoch 9/15
#    Avg Run Time (ms/batch): 4.870 AUC: 0.522 max AUC: 0.559
# >> Training model Ganomaly. Epoch 10/15
#    Avg Run Time (ms/batch): 4.898 AUC: 0.539 max AUC: 0.559
#  65% 455/703 [00:16<00:08, 28.19it/s]Reloading d net
# >> Training model Ganomaly. Epoch 11/15
#    Avg Run Time (ms/batch): 4.900 AUC: 0.529 max AUC: 0.559
# >> Training model Ganomaly. Epoch 12/15
#    Avg Run Time (ms/batch): 4.856 AUC: 0.541 max AUC: 0.559
# >> Training model Ganomaly. Epoch 13/15
#    Avg Run Time (ms/batch): 4.910 AUC: 0.528 max AUC: 0.559
# >> Training model Ganomaly. Epoch 14/15
#    Avg Run Time (ms/batch): 4.875 AUC: 0.533 max AUC: 0.559
# >> Training model Ganomaly. Epoch 15/15
#    Avg Run Time (ms/batch): 4.830 AUC: 0.531 max AUC: 0.559
# >> Training model Ganomaly.[Done]



import sklearn
import sklearn.datasets
import sklearn.datasets.kddcup99

"""KDDCUP 99 dataset.
A classic dataset for anomaly detection.
The dataset page is available from UCI Machine Learning Repository"""

import sys
import errno
from gzip import GzipFile

import logging
from io import BytesIO

import os
from os.path import exists, join

try:
    #from urllib2 import urlopen
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/kddcup99.py
dataset_kddcup99 = sklearn.datasets.kddcup99.fetch_kddcup99()

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/kddcup99.py
# use: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/kddcup99.py

# we now use: use: https://searchcode.com/codesearch/view/115660132/
# we use: http://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/sklearn/datasets/kddcup99.py

# dataset_kddcup99
print(dataset_kddcup99)



import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
#dataset = pd.read_csv('kddcup.data')
#dataset = pd.read_csv('kddcup.data_10_percent')

#dataset = pd.read_csv('/Users/dionelisnikolaos/Downloads/kddcup.data')
dataset = pd.read_csv('/Users/dionelisnikolaos/Downloads/kddcup.data_10_percent')

# we use: https://github.com/chadlimedamine/kdd-cup-99-Analysis-machine-learning-python/blob/master/kdd_binary_classification_ANN.py

#change Multi-class to binary-class
dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 41].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()

x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])

onehotencoder_1 = OneHotEncoder(categorical_features = [1])
x = onehotencoder_1.fit_transform(x).toarray()

onehotencoder_2 = OneHotEncoder(categorical_features = [4])
x = onehotencoder_2.fit_transform(x).toarray()

onehotencoder_3 = OneHotEncoder(categorical_features = [70])
x = onehotencoder_3.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split

# use: model_selection
# we use: sklearn.model_selection
#from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)



# use Keras
import keras

from keras.layers import Dense
from keras.models import Sequential

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu', input_dim = 118))

#Adding a second hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

#Adding a third hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 20)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# https://github.com/chadlimedamine/kdd-cup-99-Analysis-machine-learning-python/blob/master/kdd_binary_classification_ANN.py
# use: https://github.com/chadlimedamine/kdd-cup-99-Analysis-machine-learning-python/blob/master/kdd_binary_classification_ANN.py

# the performance of the classification model
print("the Accuracy is: "+ str((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])))

recall = cm[1,1]/(cm[0,1]+cm[1,1])

print("Recall is : "+ str(recall))
print("False Positive rate: "+ str(cm[1,0]/(cm[0,0]+cm[1,0])))

precision = cm[1,1]/(cm[1,0]+cm[1,1])

print("Precision is: "+ str(precision))
print("F-measure is: "+ str(2*((precision*recall)/(precision+recall))))

from math import log
print("Entropy is: "+ str(-precision*log(precision)))

# 244670/345814 [====================>.........] - ETA: 55s - loss: 0.0038 - acc: 0.9992
# 244720/345814 [====================>.........] - ETA: 55s - loss: 0.0038 - acc: 0.9992
# 244770/345814 [====================>.........] - ETA: 55s - loss: 0.0038 - acc: 0.9992
# 244910/345814 [====================>.........] - ETA: 54s - loss: 0.0038 - acc: 0.9992



import tensorflow as tf
import tensorflow_datasets as tfds

# use: https://www.tensorflow.org/datasets
# we now use: https://www.tensorflow.org/datasets

# tfds works in both Eager and Graph modes
tf.enable_eager_execution()

# See available datasets
print(tfds.list_builders())

# Construct a tf.data.Dataset
dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)

# we now use: https://www.tensorflow.org/datasets
dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

for features in dataset.take(1):
  image, label = features["image"], features["label"]

# https://www.tensorflow.org/datasets
# use: https://www.tensorflow.org/datasets



import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import numpy.random
import scipy.stats as ss
from sklearn.mixture import GaussianMixture

import os
import tensorflow as tf
from sklearn import metrics

# https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2
# use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2

from gluoncv.data import ImageNet

from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

from gluoncv import data, utils
from matplotlib import pyplot as plt

import scipy.io as sio
import matplotlib.pyplot as plt

# index
image_ind = 10

#train_data = sio.loadmat('train_32x32.mat')
train_data = sio.loadmat('/Users/dionelisnikolaos/Downloads/train_32x32.mat')

# SVHN Dataset
# Street View House Numbers (SVHN)

# access to the dict
x_train = train_data['X']
y_train = train_data['y']

# show sample
plt.imshow(x_train[:,:,:,image_ind])
plt.show()

print(y_train[image_ind])

image_ind = 10 # index, image index
test_data = sio.loadmat('/Users/dionelisnikolaos/Downloads/test_32x32.mat')

# access to the dict
x_test = test_data['X']
y_test = test_data['y']

# show sample
plt.imshow(x_test[:,:,:,image_ind])
plt.show()

print(y_test[image_ind])



# UCI HAR Dataset
DATASET_PATH = "/Users/dionelisnikolaos/Downloads/UCI HAR Dataset/"

TRAIN = "train/"
TEST = "test/"

# Load "X" (the neural network's training and testing inputs)
# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/README.md

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')

        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )

        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"]

X_train_signals_paths = [DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
X_test_signals_paths = [DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)

# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# Input Data
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series

n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

print('')
print(X_train.shape)
print(X_test.shape)

print('')
print(y_train.shape)
print(y_test.shape)

print('')
print(y_train)

print('')
print(y_test)

print('')
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print('')

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print('')

print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print('')



# http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html
# use: http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html

phi_i = 1/7
mu_1 = [0.0, 1.0]
mu_2 = [0.75, 0.6]
mu_3 = [1.0, 0.0]
mu_4 = [0.45, -0.8]
mu_5 = [-0.45, -0.8]
mu_6 = [-0.95, -0.2]
mu_7 = [-0.8, 0.65]

mu_total = [mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7]
sigmaSquared_i = 0.01*np.eye(2)

# we use: http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html

# v_gaussmixp([], [[0, 1]; [0.75, 0.70]; [1, 0]; [0.48, -0.8]; [-0.48, -0.8]; [-1, -0.24]; [-0.8, 0.6]], [[0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]], [1, 1, 1, 1, 1, 1, 1]')
# use: v_gaussmixp([], [[0, 1]; [0.75, 0.70]; [1, 0]; [0.48, -0.8]; [-0.48, -0.8]; [-1, -0.24]; [-0.8, 0.6]], [[0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]], [1, 1, 1, 1, 1, 1, 1]')

# find GMM probability
def prob21(x):
    prob = 0.0

    x = np.transpose(x)

    #print(x)
    #print(np.transpose(x))

    #print(phi_i)
    #print((np.linalg.det(sigmaSquared_i)))

    for i in range(7):
        #prob = prob + (phi_i * ((1 / np.sqrt(((2*np.pi)**7)*(np.linalg.det(sigmaSquared_i)))) * np.exp(-0.5*np.transpose(x-np.transpose(mu_total[i]))*(np.linalg.inv(sigmaSquared_i))*(x-np.transpose(mu_total[i])))))
        #prob = prob + (phi_i * ((1 / np.sqrt(((2*np.pi)**7)*(np.linalg.det(sigmaSquared_i)))) * np.exp(-0.5*(np.transpose(x-np.transpose(mu_total[i])))*(np.linalg.inv(sigmaSquared_i))*((x-np.transpose(mu_total[i]))))))

        #prob = prob + (phi_i * ((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))) * np.exp(-0.5 * ((x - (mu_total[i]))) * (np.linalg.inv(sigmaSquared_i)) * (np.transpose(x - (mu_total[i]))))))

        var1 = ((x - (mu_total[i])))
        var1 = np.array(var1)

        #print(mu_total[i])
        #print((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))))

        #prob = prob + (phi_i * ((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))) * np.exp(
        #    -0.5 * (((var1)) * (np.linalg.inv(sigmaSquared_i)) * ((var1.T))))))

        #prob = prob + (phi_i * ((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))) * np.exp(
        #    -0.5 * (((var1.T).dot((np.linalg.inv(sigmaSquared_i)))).dot(var1)))))

        prob = prob + (phi_i * ((1 / np.sqrt(((2 * np.pi) ** 7) * (np.linalg.det(sigmaSquared_i)))) * np.exp(
            -0.5 * (((var1).dot((np.linalg.inv(sigmaSquared_i)))).dot(var1)))))

    return prob

#prob21([1.0, 0.0])
print(prob21([1.0, 0.0]))

# http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html
# we use: http://www.ee.imperial.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gaussmixp.html

# v_gaussmixp([], [[0, 1]; [0.75, 0.70]; [1, 0]; [0.48, -0.8]; [-0.48, -0.8]; [-1, -0.24]; [-0.8, 0.6]], [[0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]; [0.05, 0.05]], [1, 1, 1, 1, 1, 1, 1]')
# v_gaussmixp([], [[0, 1]; [0.75, 0.70]; [1, 0]; [0.48, -0.8]; [-0.48, -0.8]; [-1, -0.24]; [-0.8, 0.6]], [[0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]; [0.01, 0.01]], [1, 1, 1, 1, 1, 1, 1]')

print(prob21([0.0, 1.0]))
print(prob21([0.0, 0.0]))



# numpy
import numpy as np

import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture

#X = GMMSamples(W, mu, sigma, d)
#gmm = GMM(110, covariance_type='full', random_state=0)

import numpy.random
import scipy.stats as ss

import matplotlib
import matplotlib.pyplot as plt

import os
import tensorflow as tf
from sklearn import metrics

# use: https://www.tensorflow.org/datasets

# UCI HAR Dataset
DATASET_PATH = "/Users/dionelisnikolaos/Downloads/UCI HAR Dataset/"

# we use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2

TRAIN = "train/"
TEST = "test/"

# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')

        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )

        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"]

X_train_signals_paths = [DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]

X_test_signals_paths = [DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)

# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# Input Data
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series

n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

print('')
print(X_train.shape)
print(X_test.shape)

print('')
print(y_train.shape)
print(y_test.shape)



# LSTM Neural Network's internal structure
n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)

# Training
learning_rate = 0.0025

lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset

batch_size = 1500
display_iter = 30000  # To show test set accuracy during training

print('')
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))

print('')
print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")

print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print('')



# use LSTM
def LSTM_RNN(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.

    # Note, some code of this notebook is inspired from an slightly different
    # RNN architecture used on another dataset, some of the credits goes to
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size

    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)

    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size

    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s

def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes

    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))

    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/README.md
# use: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/README.md



mean = [0, 0]

# diagonal covariance
cov = [[1, 0], [0, 100]]

import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 5000).T

plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()

n = 10000
numpy.random.seed(0x5eed)

# Parameters of the mixture components
norm_params = np.array([[5, 1],
                        [1, 1.3],
                        [9, 1.3]])

n_components = norm_params.shape[0]

# Weight of each component, in this case all of them are 1/3
weights = np.ones(n_components, dtype=np.float64) / float(n_components)

# A stream of indices from which to choose the component
mixture_idx = numpy.random.choice(n_components, size=n, replace=True, p=weights)

# y is the mixture sample
y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                   dtype=np.float64)

# Theoretical PDF plotting -- generate the x and y plotting positions
xs = np.linspace(y.min(), y.max(), 200)
ys = np.zeros_like(xs)

for (l, s), w in zip(norm_params, weights):
    ys += ss.norm.pdf(xs, loc=l, scale=s) * w

plt.plot(xs, ys)
plt.hist(y, normed=True, bins="fd")

plt.xlabel("x")
plt.ylabel("f(x)")

plt.show()



# Generate synthetic data
N,D = 1000, 2 # number of points and dimenstinality

if D == 2:
    #set gaussian ceters and covariances in 2D
    means = np.array([[0.5, 0.0],
                      [0, 0],
                      [-0.5, -0.5],
                      [-0.8, 0.3]])
    covs = np.array([np.diag([0.01, 0.01]),
                     np.diag([0.025, 0.01]),
                     np.diag([0.01, 0.025]),
                     np.diag([0.01, 0.01])])
elif D == 3:
    # set gaussian ceters and covariances in 3D
    means = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [-0.5, -0.5, -0.5],
                      [-0.8, 0.3, 0.4]])
    covs = np.array([np.diag([0.01, 0.01, 0.03]),
                     np.diag([0.08, 0.01, 0.01]),
                     np.diag([0.01, 0.05, 0.01]),
                     np.diag([0.03, 0.07, 0.01])])
n_gaussians = means.shape[0]

points = []
for i in range(len(means)):
    x = np.random.multivariate_normal(means[i], covs[i], N )
    points.append(x)

points = np.concatenate(points)

#fit the gaussian model
gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
gmm.fit(points)

# use numpy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mpl_toolkits.mplot3d import Axes3D

import os
import matplotlib.cm as cmx

def visualize_3d_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 3D
    Input:
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))

    # Visualize data
    fig = plt.figure(figsize=(8, 8))

    axes = fig.add_subplot(111, projection='3d')

    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set_zlim([-1, 1])

    plt.set_cmap('Set1')

    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))

    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)

        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title('3D GMM')

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    axes.view_init(35.246, 45)

    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/3D_GMM_demonstration.png', dpi=100, format='png')

    plt.show()

def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input:
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    pi = np.pi
    cos = np.cos
    sin = np.sin

    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]

    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]

    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')

    c = cmap.to_rgba(w)
    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)
    return ax

def visualize_2D_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 2D

    Input:
        points: N X 2, sampled points
        w: n_gaussians, gmm weights
        mu: 2 X n_gaussians, gmm means
        stdev: 2 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))

    # Visualize data
    fig = plt.figure(figsize=(8, 8))

    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])

    plt.set_cmap('Set1')

    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))

    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)

        plt.scatter(points[idx, 0], points[idx, 1], alpha=0.3, c=colors[i])

        for j in range(8):
            axes.add_patch(
                patches.Ellipse(mu[:, i], width=(j+1) * stdev[0, i], height=(j+1) *  stdev[1, i], fill=False, color=[0.0, 0.0, 1.0, 1.0/(0.5*j+1)]))
        plt.title('GMM')

    plt.xlabel('X')
    plt.ylabel('Y')

    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/2D_GMM_demonstration.png', dpi=100, format='png')

    plt.show()

#visualize
if D == 2:
    visualize_2D_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
elif D == 3:
    visualize_3d_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)



#Caltech-101 Dataset
#CIFAR-10 Dataset, CIFAR-100 Dataset

# we use Sequential
from keras.models import Sequential
from keras.layers import Dense

# use dropout
from keras.layers import Dropout
from keras.layers import Flatten

from keras.constraints import maxnorm
from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
from keras import initializers

from keras import backend as K

K.set_image_dim_ordering('th')

import os
import numpy as np

import scipy.io
import scipy.misc

# matplotlib
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf

def imread(path):
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2:
		img = np.transpose(np.array([img, img, img]), (2, 0, 1))
	return img

#cwd = os.getcwd()
#path = cwd + "/101_ObjectCategories"

#path = "/101_ObjectCategories"
path = "/Users/dionelisnikolaos/Downloads/101_ObjectCategories"

#CIFAR-10 Dataset
#Caltech-101 Dataset
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)

imgs = []
labels = []

print('')
#print(categories)
print(categories[1:])

print('')
categories = categories[1:]

# LOAD ALL IMAGES
for i, category in enumerate(categories):
	iter = 0

	for f in os.listdir(path + "/" + category):
		if iter == 0:
			ext = os.path.splitext(f)[1]

			if ext.lower() not in valid_exts:
				continue

			fullpath = os.path.join(path + "/" + category, f)

			img = scipy.misc.imresize(imread(fullpath), [128, 128, 3])
			img = img.astype('float32')

			img[:, :, 0] -= 123.68
			img[:, :, 1] -= 116.78
			img[:, :, 2] -= 103.94

			imgs.append(img)  # NORMALIZE IMAGE

			label_curr = i
			labels.append(label_curr)

	# iter = (iter+1)%10;

print("Num imgs: %d" % (len(imgs)))
print("Num labels: %d" % (len(labels)))

print(ncategories)

seed = 7
np.random.seed(seed)

# use pandas
import pandas as pd

# use sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.1)

X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)

X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)

print("Num train_imgs: %d" % (len(X_train)))
print("Num test_imgs: %d" % (len(X_test)))

# # one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
print(y_test.shape)

print(X_train[1, 1, 1, :])
print(y_train[1])

# normalize inputs from 0-255 to 0.0-1.0
print(X_train.shape)
print(X_test.shape)

X_train = X_train.transpose(0, 3, 1, 2)
X_test = X_test.transpose(0, 3, 1, 2)

print(X_train.shape)
print(X_test.shape)

# we use scipy
import scipy.io as sio

data = {}

data['categories'] = categories

data['X_train'] = X_train
data['y_train'] = y_train

data['X_test'] = X_test
data['y_test'] = y_test

sio.savemat('caltech_del.mat', data)



# CIFAR-10 Dataset
# CNN model for CIFAR-10

# numpy
import numpy

# CIFAR-10 dataset
from keras.datasets import cifar10

# Sequential
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Dropout

from keras.layers import Flatten
from keras.constraints import maxnorm

from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras import backend as K
from keras.utils import np_utils

K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

# we now create the model
# we use: https://github.com/acht7111020/CNN_object_classification

# use Sequential
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))

model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 50
lrate = 0.01

decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print('')
print(model.summary())
print('')

# https://github.com/acht7111020/CNN_object_classification
# use: https://github.com/acht7111020/CNN_object_classification



#Caltech-101 Dataset
#CIFAR-10 and CIFAR-100 Datasets

# we use Sequential
from keras.models import Sequential
from keras.layers import Dense

# use dropout
from keras.layers import Dropout
from keras.layers import Flatten

from keras.constraints import maxnorm
from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
from keras import initializers

from keras import backend as K

K.set_image_dim_ordering('th')

import os
import numpy as np

import scipy.io
import scipy.misc

# matplotlib
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf

def imread(path):
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2:
		img = np.transpose(np.array([img, img, img]), (2, 0, 1))
	return img

#cwd = os.getcwd()
#path = cwd + "/101_ObjectCategories"

#path = "/101_ObjectCategories"
path = "/Users/dionelisnikolaos/Downloads/101_ObjectCategories"

#CIFAR-10 Dataset
#Caltech-101 Dataset

#CIFAR-10 Dataset
#CIFAR-100 Dataset
#Caltech-101 Dataset

valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)

imgs = []
labels = []

print('')

#print(categories)
print(categories[1:])

print('')
categories = categories[1:]

# LOAD ALL IMAGES
for i, category in enumerate(categories):
	iter = 0

	for f in os.listdir(path + "/" + category):
		if iter == 0:
			ext = os.path.splitext(f)[1]

			if ext.lower() not in valid_exts:
				continue

			fullpath = os.path.join(path + "/" + category, f)

			img = scipy.misc.imresize(imread(fullpath), [128, 128, 3])
			img = img.astype('float32')

			img[:, :, 0] -= 123.68
			img[:, :, 1] -= 116.78
			img[:, :, 2] -= 103.94

			imgs.append(img)  # NORMALIZE IMAGE

			label_curr = i
			labels.append(label_curr)

	# iter = (iter+1)%10;

print("Num imgs: %d" % (len(imgs)))
print("Num labels: %d" % (len(labels)))

print(ncategories)

seed = 7
np.random.seed(seed)

# use pandas
import pandas as pd

# use sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.1)

X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)

X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)

print("Num train_imgs: %d" % (len(X_train)))
print("Num test_imgs: %d" % (len(X_test)))

# # one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
print(y_test.shape)

print(X_train[1, 1, 1, :])
print(y_train[1])

# normalize inputs from 0-255 to 0.0-1.0
print(X_train.shape)
print(X_test.shape)

X_train = X_train.transpose(0, 3, 1, 2)
X_test = X_test.transpose(0, 3, 1, 2)

print(X_train.shape)
print(X_test.shape)

# we use scipy
import scipy.io as sio

data = {}
data['categories'] = categories

data['X_train'] = X_train
data['y_train'] = y_train

data['X_test'] = X_test
data['y_test'] = y_test

sio.savemat('caltech_del.mat', data)



from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# Create the model
model = Sequential()

# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=(3, 128, 128), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile mode
epochs = 300
lrate = 0.0001

decay = lrate / epochs

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = SGD(lr=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

np.random.seed(seed)

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
				 epochs=epochs, batch_size=56, shuffle=True, callbacks=[earlyStopping])

# hist = model.load_weights('./64.15/model.h5');

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1] * 100))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.legend(['train', 'test'])
plt.title('loss')

plt.savefig("loss7.png", dpi=300, format="png")

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.legend(['train', 'test'])
plt.title('accuracy')

plt.savefig("accuracy7.png", dpi=300, format="png")

model_json = model.to_json()
with open("model7.json", "w") as json_file:
	json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model7.h5")
print("Saved model to disk")



#import numpy
import numpy as np

# https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2
# use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2

# we use: https://skymind.ai/wiki/open-datasets
# use: http://people.csail.mit.edu/yalesong/cvpr12/

from keras.datasets import mnist
((trainX, trainY), (testX, testY)) = mnist.load_data()

print(trainX.shape)
print(testX.shape)

from keras.datasets import fashion_mnist
((trainX2, trainY2), (testX2, testY2)) = fashion_mnist.load_data()

print(trainX2.shape)
print(testX2.shape)
print('')

from keras.datasets import imdb
((trainX3, trainY3), (testX3, testY3)) = imdb.load_data()

print(trainX3.shape)
print(testX3.shape)
print('')

from keras.datasets import cifar10
((trainX4, trainY4), (testX4, testY4)) = cifar10.load_data()

print(trainX4.shape)
print(testX4.shape)

from keras.datasets import cifar100
((trainX5, trainY5), (testX5, testY5)) = cifar100.load_data()

print(trainX5.shape)
print(testX5.shape)
print('')

# use: https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d
# we now use: https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d

from keras.datasets import reuters
((trainX6, trainY6), (testX6, testY6)) = reuters.load_data()

print(trainX6.shape)
print(testX6.shape)

from keras.datasets import boston_housing
((trainX7, trainY7), (testX7, testY7)) = boston_housing.load_data()

print(trainX7.shape)
print(testX7.shape)
print('')

# use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2
# we use: https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#from sklearn import datasets
#from sklearn import datasets2

import sklearn
#from sklearn.datasets2 import kddcup99

#import sklearn.datasets2
#import sklearn.datasets

#dataset_boston = datasets.load_boston()
#dataset_boston = datasets2.load_boston()

#dataset_kddcup99 = datasets2.load_digits()



# use .io
import scipy.io

#mat2 = scipy.io.loadmat('NATOPS6.mat')
mat2 = scipy.io.loadmat('/Users/dionelisnikolaos/Downloads/NATOPS6.mat')

# NATOPS6.mat
print(mat2)

#mat = scipy.io.loadmat('thyroid.mat')
mat = scipy.io.loadmat('/Users/dionelisnikolaos/Downloads/thyroid.mat')

# thyroid.mat
print(mat)



# usenet_recurrent3.3.data
# we use: usenet_recurrent3.3.data

# use pandas
import pandas as pd

# numpy
import numpy

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss

from sklearn.model_selection import train_test_split

data_dir = "/Users/dionelisnikolaos/Downloads/"
raw_data_filename = data_dir + "usenet_recurrent3.3.data"

#raw_data_filename = "/Users/dionelisnikolaos/Downloads/usenet_recurrent3.3.data"

# raw_data_filename = "/Users/dionelisnikolaos/Downloads/usenet_recurrent3.3.data"
# use: raw_data_filename = "/Users/dionelisnikolaos/Downloads/usenet_recurrent3.3.data"

print ("Loading raw data")
raw_data = pd.read_csv(raw_data_filename, header=None)

print ("Transforming data")

# Categorize columns: "protocol", "service", "flag", "attack_type"
raw_data[1], protocols= pd.factorize(raw_data[1])
raw_data[2], services = pd.factorize(raw_data[2])

raw_data[3], flags    = pd.factorize(raw_data[3])
raw_data[41], attacks = pd.factorize(raw_data[41])

# separate features (columns 1..40) and label (column 41)
features= raw_data.iloc[:,:raw_data.shape[1]-1]
labels= raw_data.iloc[:,raw_data.shape[1]-1:]

# convert them into numpy arrays
#features= numpy.array(features)

#labels= numpy.array(labels).ravel() # this becomes an 'horizontal' array
labels= labels.values.ravel() # this becomes a 'horizontal' array

# Separate data in train set and test set
df= pd.DataFrame(features)

# create training and testing vars
# Note: train_size + test_size < 1.0 means we are subsampling

# Use small numbers for slow classifiers, as KNN, Radius, SVC,...
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2)

print('')

print ("X_train, y_train:", X_train.shape, y_train.shape)
print ("X_test, y_test:", X_test.shape, y_test.shape)

print('')

print(X_train.shape)
print(y_train.shape)

print('')

print(X_train.shape)
print(X_test.shape)

print('')



# use matplotlib
import matplotlib.pyplot as plt

# we use: https://skymind.ai/wiki/open-datasets
# use: http://people.csail.mit.edu/yalesong/cvpr12/

from csv import reader

# Load a CSV file
def load_csv(filename):
    file = open(filename, "r")

    lines = reader(file)
    dataset = list(lines)

    return dataset

dataset = load_csv('/Users/dionelisnikolaos/Downloads/ann-train.data.txt')

# Load dataset

filename = '/Users/dionelisnikolaos/Downloads/ann-train.data.txt'
#print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))

#file = open(filename, 'r')
#for line in file:
#    print (line,)

text_file = open(filename, "r")
lines = text_file.read().split(' ')

#print(lines)

list_of_lists = []
with open(filename) as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]

        # in alternative, if you need to use the file content as numbers
        # inner_list = [int(elt.strip()) for elt in line.split(',')]
        list_of_lists.append(inner_list)

print(list_of_lists)



import pandas as pd
import numpy

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss

from sklearn.model_selection import train_test_split

#data_dir="./datasets/KDD-CUP-99/"
#data_dir="./"

data_dir = "/Users/dionelisnikolaos/Downloads/"
raw_data_filename = data_dir + "kddcup.data"

#raw_data_filename = "/Users/dionelisnikolaos/Downloads/kddcup.data"

print ("Loading raw data")

raw_data = pd.read_csv(raw_data_filename, header=None)

print ("Transforming data")

# Categorize columns: "protocol", "service", "flag", "attack_type"
raw_data[1], protocols= pd.factorize(raw_data[1])
raw_data[2], services = pd.factorize(raw_data[2])

raw_data[3], flags    = pd.factorize(raw_data[3])
raw_data[41], attacks = pd.factorize(raw_data[41])

# separate features (columns 1..40) and label (column 41)
features= raw_data.iloc[:,:raw_data.shape[1]-1]
labels= raw_data.iloc[:,raw_data.shape[1]-1:]

# convert them into numpy arrays
#features= numpy.array(features)

# this becomes an 'horizontal' array
#labels= numpy.array(labels).ravel()

# this becomes a 'horizontal' array
labels= labels.values.ravel()

# Separate data in train set and test set
df= pd.DataFrame(features)

# create training and testing vars
# Note: train_size + test_size < 1.0 means we are subsampling

# Use small numbers for slow classifiers, as KNN, Radius, SVC,...
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2)

print('')
print ("X_train, y_train:", X_train.shape, y_train.shape)
print ("X_test, y_test:", X_test.shape, y_test.shape)

print('')
print(X_train.shape)
print(y_train.shape)

print('')
print(X_train.shape)
print(X_test.shape)

print('')

# Training, choose model by commenting/uncommenting clf=
print ("Training model")

clf= RandomForestClassifier(n_jobs=-1, random_state=3, n_estimators=102)#, max_features=0.8, min_samples_leaf=3, n_estimators=500, min_samples_split=3, random_state=10, verbose=1)
#clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, presort=False)

trained_model= clf.fit(X_train, y_train)

print ("Score: ", trained_model.score(X_train, y_train))

# Predicting
print ("Predicting")
y_pred = clf.predict(X_test)

print ("Computing performance metrics")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)

print ("Confusion matrix:\n", results)
print ("Error: ", error)

# KDD99 Dataset
# use: https://github.com/ghuecas/kdd99ml

# https://github.com/ghuecas/kdd99ml
# we use: https://github.com/ghuecas/kdd99ml



import json
import datetime

import os
import numpy as np

# make keras deterministic
#np.random.seed(42)

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.callbacks import CallbackList, ModelCheckpoint
from keras.regularizers import l2

import os

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#from keras.applications.inception_v3 import InceptionV3
#base_model = InceptionV3(weights='imagenet', include_top=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

num_train_images =  1500
num_test_images = 100

#-------------------
# organize imports
#-------------------
import numpy as np

import os
import h5py

import glob
import cv2

# we use opencv-python
import cv2

# we use keras
from keras.preprocessing import image

#------------------------
# dataset pre-processing
#------------------------
#train_path   = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\train"
#test_path    = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test"

#train_path   = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\train"
train_path   = "/Users/dionelisnikolaos/Downloads/dataset/train"

#test_path    = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test"
test_path    = "/Users/dionelisnikolaos/Downloads/dataset/test"

train_labels = os.listdir(train_path)
test_labels  = os.listdir(test_path)

# tunable parameters
image_size       = (64, 64)

num_train_images = 1500
num_test_images  = 100

num_channels     = 3

# train_x dimension = {(64*64*3), 1500}
# train_y dimension = {1, 1500}
# test_x dimension  = {(64*64*3), 100}
# test_y dimension  = {1, 100}

train_x = np.zeros(((image_size[0]*image_size[1]*num_channels), num_train_images))
train_y = np.zeros((1, num_train_images))

test_x  = np.zeros(((image_size[0]*image_size[1]*num_channels), num_test_images))
test_y  = np.zeros((1, num_test_images))

#----------------
# TRAIN dataset
#----------------
count = 0
num_label = 0

for i, label in enumerate(train_labels):
	cur_path = train_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)

		x   = image.img_to_array(img)

		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)

		train_x[:,count] = x
		train_y[:,count] = num_label

		count += 1
	num_label += 1

#--------------
# TEST dataset
#--------------
count = 0
num_label = 0

for i, label in enumerate(test_labels):
	cur_path = test_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)

		x   = image.img_to_array(img)

		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)

		test_x[:,count] = x
		test_y[:,count] = num_label

		count += 1
	num_label += 1

#------------------
# standardization
#------------------
train_x = train_x/255.
test_x  = test_x/255.

print ("train_labels : " + str(train_labels))

print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))

print ("test_x shape : " + str(test_x.shape))
print ("test_y shape : " + str(test_y.shape))

print('')

# train_x and test_x
print(train_x.shape)
print(test_x.shape)

# https://gogul09.github.io/software/neural-nets-logistic-regression
# use: https://gogul09.github.io/software/neural-nets-logistic-regression

# save using h5py
h5_train = h5py.File("train_x.h5", 'w')
h5_train.create_dataset("data_train", data=np.array(train_x))

h5_train.close()

h5_test = h5py.File("test_x.h5", 'w')
h5_test.create_dataset("data_test", data=np.array(test_x))

h5_test.close()

def sigmoid(z):
	return (1/(1+np.exp(-z)))

def init_params(dimension):
	w = np.zeros((dimension, 1))
	b = 0
	return w, b

def propagate(w, b, X, Y):
	# num of training samples
	m = X.shape[1]

	# forward pass
	A    = sigmoid(np.dot(w.T,X) + b)
	cost = (-1/m)*(np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A))))

	# back propagation
	dw = (1/m)*(np.dot(X, (A-Y).T))
	db = (1/m)*(np.sum(A-Y))

	cost = np.squeeze(cost)

	# gradient dictionary
	grads = {"dw": dw, "db": db}

	return grads, cost

def optimize(w, b, X, Y, epochs, lr):
	costs = []
	for i in range(epochs):
		# calculate gradients
		grads, cost = propagate(w, b, X, Y)

		# get gradients
		dw = grads["dw"]
		db = grads["db"]

		# update rule
		w = w - (lr*dw)
		b = b - (lr*db)

		if i % 100 == 0:
			costs.append(cost)
			print ("cost after %i epochs: %f" %(i, cost))

	# param dict
	params = {"w": w, "b": b}

	# gradient dict
	grads  = {"dw": dw, "db": db}

	return params, grads, costs

def predict(w, b, X):
	m = X.shape[1]

	Y_predict = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(np.dot(w.T, X) + b)

	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_predict[0, i] = 0
		else:
			Y_predict[0,i]  = 1

	return Y_predict

def predict_image(w, b, X):
	Y_predict = None

	w = w.reshape(X.shape[0], 1)
	A = sigmoid(np.dot(w.T, X) + b)

	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_predict = 0
		else:
			Y_predict = 1

	return Y_predict

def model(X_train, Y_train, X_test, Y_test, epochs, lr):
	w, b = init_params(X_train.shape[0])
	params, grads, costs = optimize(w, b, X_train, Y_train, epochs, lr)

	w = params["w"]
	b = params["b"]

	Y_predict_train = predict(w, b, X_train)
	Y_predict_test  = predict(w, b, X_test)

	print ("train_accuracy: {} %".format(100-np.mean(np.abs(Y_predict_train - Y_train)) * 100))
	print ("test_accuracy : {} %".format(100-np.mean(np.abs(Y_predict_test  - Y_test)) * 100))

	log_reg_model = {"costs": costs,
				     "Y_predict_test": Y_predict_test,
					 "Y_predict_train" : Y_predict_train,
					 "w" : w,
					 "b" : b,
					 "learning_rate" : lr,
					 "epochs": epochs}

	return log_reg_model

# we use: https://gogul09.github.io/software/neural-nets-logistic-regression

epochs = 10

# lr, learning rate, step size
lr = 0.0003

# activate the logistic regression model
myModel = model(train_x, train_y, test_x, test_y, epochs, lr)

#test_img_paths = ["G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0723.jpg",
#                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0713.jpg",
#                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0782.jpg",
#                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0799.jpg",
#                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\test_1.jpg"]

# https://gogul09.github.io/software/neural-nets-logistic-regression
# use: https://gogul09.github.io/software/neural-nets-logistic-regression

test_img_paths = ["/Users/dionelisnikolaos/Downloads/dataset/test/airplane/image_0763.jpg",
                  "/Users/dionelisnikolaos/Downloads/dataset/test/airplane/image_0753.jpg",
                  "/Users/dionelisnikolaos/Downloads/dataset/test/bike/image_0782.jpg",
                  "/Users/dionelisnikolaos/Downloads/dataset/test/bike/image_0799.jpg",
                  "/Users/dionelisnikolaos/Downloads/dataset/test/bike/image_0751.jpg"]

for test_img_path in test_img_paths:
	img_to_show    = cv2.imread(test_img_path, -1)
	img            = image.load_img(test_img_path, target_size=image_size)

	x              = image.img_to_array(img)
	x              = x.flatten()
	x              = np.expand_dims(x, axis=1)

	predict        = predict_image(myModel["w"], myModel["b"], x)
	predict_label  = ""

	if predict == 0:
		predict_label = "airplane"
	else:
		predict_label = "bike"

	# display the test image and the predicted label
	cv2.putText(img_to_show, predict_label, (30,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("test_image", img_to_show)

	key = cv2.waitKey(0) & 0xFF

	if (key == 27):
		cv2.destroyAllWindows()



import keras
import keras.datasets

# use datasets
import keras.datasets

from keras.datasets import cifar10
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()



from keras.datasets import fashion_mnist
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# set the matplotlib backend so figures can be saved in the background
import matplotlib

# import the necessary packages
from sklearn.metrics import classification_report
from keras.optimizers import SGD

# use Fashion-MNIST
from keras.datasets import fashion_mnist

from keras.utils import np_utils
from keras import backend as K

#from imutils import build_montages
import numpy as np

# use matplotlib
import matplotlib.pyplot as plt

#image_index = 7777
image_index = 777

# ((trainX, trainY), (testX, testY))
# (x_train, y_train), (x_test, y_test)
y_train = trainY
x_train = trainX

# ((trainX, trainY), (testX, testY))
# (x_train, y_train), (x_test, y_test)
y_test = testY
x_test = testX

print(trainX.shape)
print(trainY.shape)

print(testX.shape)
print(testY.shape)

print(y_train[image_index].shape)
print(x_train[image_index].shape)

print(y_train[image_index])

plt.imshow(x_train[image_index], cmap='Greys')
#plt.imshow(x_train[image_index])

plt.pause(2)

#x_train.shape
print(x_train.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# we define the input shape
input_shape = (28, 28, 1)

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

# import the necessary packages
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# import the necessary packages
from keras.layers.core import Activation
from keras.layers.core import Flatten

# use dropout
from keras.layers.core import Dropout

from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()

        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

            chanDim = 1

            # first CONV => RELU => CONV => RELU => POOL layer set
            model.add(Conv2D(32, (3, 3), padding="same",
                             input_shape=inputShape))

            model.add(Activation("relu"))

            model.add(BatchNormalization(axis=chanDim))
            model.add(Conv2D(32, (3, 3), padding="same"))

            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # second CONV => RELU => CONV => RELU => POOL layer set
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation("relu"))

            model.add(BatchNormalization(axis=chanDim))
            model.add(Conv2D(64, (3, 3), padding="same"))

            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # first (and only) set of FC => RELU layers
            model.add(Flatten())
            model.add(Dense(512))

            model.add(Activation("relu"))
            model.add(BatchNormalization())

            model.add(Dropout(0.5))

            # softmax classifier
            model.add(Dense(classes))
            model.add(Activation("softmax"))

            # return the constructed network architecture
            return model



# use numpy
import numpy as np

#matplotlib inline
import matplotlib.pyplot as plt

# use tensorflow
import tensorflow as tf

# we use the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
# use: https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

import matplotlib.pyplot as plt
image_index = 7777

# The label is 8
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')

#plt.pause(5)
plt.pause(2)

#x_train.shape
print(x_train.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# we define the input shape
input_shape = (28, 28, 1)

# the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)

print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])



# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 2D arrays for fully connected layers
model.add(Flatten())

model.add(Dense(128, activation=tf.nn.relu))

model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ADAM, adaptive momentum
# we use the Adam optimizer

# fit the model
#model.fit(x=x_train,y=y_train, epochs=10)

#model.fit(x=x_train,y=y_train, epochs=10)
model.fit(x=x_train,y=y_train, epochs=8)

# evaluate the model
model.evaluate(x_test, y_test)

# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

# use index 4444
image_index = 4444

plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')

#plt.pause(5)
plt.pause(2)

#pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))

print(pred.argmax())



# Deep Generative Models
# GANs and VAEs, Generative Models

# random noise
# from random noise to a tensor

# We use batch normalisation.
# GANs are very difficult to train. Super-deep models. This is why we use batch normalisation.

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb



# Anomaly detection (AD)
# Unsupervised machine learning

# GANs for super-resolution
# Generative Adversarial Networks, GANs

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



import torch
import torchvision

from torchvision import datasets, transforms
#from torchvision import transforms, datasets

# use matplotlib
import matplotlib.pyplot as plt

# use nn.functional
import torch.nn.functional as F

#import matplotlib.pyplot as plt
#batch_size = 128

# download the training dataset
#train_data = datasets.FashionMNIST(root='fashiondata/',
#                                   transform=transforms.ToTensor(),
#                                   train=True,
#                                   download=True)

# we create the train data loader
#train_loader = torch.utils.data.DataLoader(train_data,
#                                           shuffle=True,
#                                           batch_size=batch_size)

# define the batch size
batch_size = 100

train_data = datasets.FashionMNIST(root='fashiondata/',
                                 transform=transforms.ToTensor(),
                                 train=True,
                                 download=True
                                 )

train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True
                                           )

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# class for D and G
# we train the discriminator and the generator

# we make the discriminator
class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)  # 1x28x28-> 64x14x14
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 64x14x14-> 128x7x7

        self.dense1 = torch.nn.Linear(128 * 7 * 7, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))).view(-1, 128 * 7 * 7)

        # use sigmoid for the output layer
        x = F.sigmoid(self.dense1(x))

        return x

# this was for the discriminator
# we now do the same for the generator

# Generator G
class generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(256, 1024)
        self.dense3 = torch.nn.Linear(1024, 128 * 7 * 7)

        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 128x7x7 -> 64x14x14
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)  # 64x14x14 -> 1x28x28

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.bn3 = torch.nn.BatchNorm1d(128 * 7 * 7)
        self.bn4 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = F.relu(self.bn3(self.dense3(x))).view(-1, 128, 7, 7)

        x = F.relu(self.bn4(self.uconv1(x)))

        x = F.sigmoid(self.uconv2(x))
        return x

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb
# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# instantiate the model
d = discriminator()
g = generator()

# training hyperparameters
#epochs = 100
epochs = 10

# learning rate
#dlr = 0.0003
#glr = 0.0003

dlr = 0.003
glr = 0.003

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

def train(epochs, glr, dlr):
    g_losses = []
    d_losses = []

    for epoch in range(epochs):

        # iteratre over mini-batches
        for batch_idx, (real_images, _) in enumerate(train_samples):

            z = torch.randn(batch_size, 128)  # generate random latent variable to generate images from
            generated_images = g.forward(z)  # generate images

            gen_pred = d.forward(generated_images)  # prediction of discriminator on generated batch
            real_pred = d.forward(real_images)  # prediction of discriminator on real batch

            dcost = -torch.sum(torch.log(real_pred)) - torch.sum(torch.log(1 - gen_pred))  # cost of discriminator
            gcost = -torch.sum(torch.log(gen_pred)) / batch_size  # cost of generator

            # train discriminator
            d_optimizer.zero_grad()
            dcost.backward(retain_graph=True)  # retain the computational graph so we can train generator after
            d_optimizer.step()

            # train generator
            g_optimizer.zero_grad()

            gcost.backward()
            g_optimizer.step()

            # give us an example of a generated image after every 10000 images produced
            #if batch_idx * batch_size % 10000 == 0:

            # give us an example of a generated image after every 20 images produced
            if batch_idx % 20 == 0:
                g.eval()  # put in evaluation mode
                noise_input = torch.randn(1, 128)
                generated_image = g.forward(noise_input)

                generated_img.imshow(generated_image.detach().squeeze(), cmap='gray_r')

                # pause for some seconds
                plt.pause(5)

                # put back into training mode
                g.train()

            dcost /= batch_size
            gcost /= batch_size

            print('Epoch: ', epoch, 'Batch idx:', batch_idx, '\tDisciminator cost: ', dcost.item(),
                  '\tGenerator cost: ', gcost.item())

            dcosts.append(dcost)
            gcosts.append(gcost)

            loss_ax.plot(dcosts, 'b')
            loss_ax.plot(gcosts, 'r')

            fig.canvas.draw()

#print(torch.__version__)
train(epochs, glr, dlr)

# Epoch:  0 Batch idx: 0 	Disciminator cost:  1.3832124471664429 	Generator cost:  0.006555716972798109
# Epoch:  0 Batch idx: 1 	Disciminator cost:  1.0811840295791626 	Generator cost:  0.008780254982411861
# Epoch:  0 Batch idx: 2 	Disciminator cost:  0.8481155633926392 	Generator cost:  0.011281056329607964
# Epoch:  0 Batch idx: 3 	Disciminator cost:  0.6556042432785034 	Generator cost:  0.013879001140594482
# Epoch:  0 Batch idx: 4 	Disciminator cost:  0.5069876909255981 	Generator cost:  0.016225570812821388
# Epoch:  0 Batch idx: 5 	Disciminator cost:  0.4130948781967163 	Generator cost:  0.018286770209670067
# Epoch:  0 Batch idx: 6 	Disciminator cost:  0.33445805311203003 	Generator cost:  0.02015063539147377
# Epoch:  0 Batch idx: 7 	Disciminator cost:  0.279323011636734 	Generator cost:  0.021849267184734344

# Epoch:  0 Batch idx: 41 	Disciminator cost:  0.10074597597122192 	Generator cost:  0.03721988573670387
# Epoch:  0 Batch idx: 42 	Disciminator cost:  0.07906078547239304 	Generator cost:  0.04363853484392166

# Epoch:  0 Batch idx: 118 	Disciminator cost:  0.010556117631494999 	Generator cost:  0.06929603219032288
# Epoch:  0 Batch idx: 119 	Disciminator cost:  0.017774969339370728 	Generator cost:  0.07270769774913788

# Epoch:  0 Batch idx: 447 	Disciminator cost:  0.12328958511352539 	Generator cost:  0.03817861154675484
# Epoch:  0 Batch idx: 448 	Disciminator cost:  0.06865841150283813 	Generator cost:  0.03938257694244385

# generate random latent variable to generate images
z = torch.randn(batch_size, 128)

# generate images
im = g.forward(z)
# use "forward(.)"

plt.imshow(im)

