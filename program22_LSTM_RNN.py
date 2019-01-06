# we use: F. Chollet's book "Deep learning with Python", Keras

# we use: http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf
# use: Keras and F. Chollet's book "Deep learning with Python"

# LSTM = Long Short Term memory, LSTM recurrent neural networks (RNNs)
# GRU = Gated Recurrent Unit, GRU RNNs

# we use: LSTM RNNs and GRU RNNs

# LSTM RNNs are used in (https://research.google.com/pubs/archive/44312.pdf)
# GRU RNNs are used in (https://pdfs.semanticscholar.org/702e/aa99bcb366d08d7f450ed7e354f9f6920b23.pdf)

# GRU RNNs along with mixture density networks (MDNs) are used in (https://pdfs.semanticscholar.org/702e/aa99bcb366d08d7f450ed7e354f9f6920b23.pdf)

# we use: LSTM RNNs and GRU RNNs
# LSTM RNNs and GRU RNNs are better than (vanilla) RNNs

# ﻿https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf



# we use the Kapre library
# use: https://github.com/keunwoochoi/kapre

# Kapre and Keras can be used together

# Deep neural networks (DNNs) are used in one of the papers in (https://drive.google.com/drive/folders/1GSMA7KPnJQ0LFBu3F3t7VmybFtAn580U)
# https://www.commsp.ee.ic.ac.uk/~sap/people-nikolaos-dionelis/

# we use the terminal and the command line to download files

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

# use numpy
import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))

# we now parse the data
# we convert the 420551 lines of data into a Numpy array

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values



# we use pyplot
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
# we now prepare the data

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
# we create a baseline algorithm

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

# Big Data, AI, Machine Learning, Data Science
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf



# we use a GRU
# we create a bidirectional GRU RNN

from keras.models import Sequential
from keras import layers

# use RMSprop
from keras.optimizers import RMSprop

model = Sequential()

model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, \
                              validation_data=val_gen, validation_steps=val_steps)

# we use Keras
# use: http://crcv.ucf.edu/courses/CAP6412/Spring2018/KerasTutorial.pdf

# http://zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf

# website: http://crcv.ucf.edu/courses/CAP6412/Spring2018/KerasTutorial.pdf
# use: http://zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf

# AI: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf



# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# binary search => requires a sorted list
# we use: http://interactivepython.org/runestone/static/pythonds/Recursion/toctree.html

# https://www.w3resource.com/c-programming-exercises/recursion/index.php

# binary search
def binarySearch(list1, item1):
    upper = len(list1) - 1
    lower = 0

    # define Boolean found
    found = False

    while not found and upper >= lower:
        mid = (upper + lower) // 2
        # use "// 2" integer division

        if list1[mid] == item1:
            found = True
        elif list1[mid] < item1:
            lower = mid + 1
        else:
            upper = mid - 1

        # if upper < lower:
        #    break

    # return the Boolean found
    return found

list1 = [4, 5, 6, 7]
print(binarySearch(list1, 6))

# def binarySearch_rec(list1, item1, upper=len(list1)-1, lower=0):
def binarySearch_rec(list1, item1, upper, lower=0):
    mid = (upper + lower) // 2

    if upper < lower:
        return False

    if item1 == list1[mid]:
        return True

    return binarySearch_rec(list1, item1, upper, mid + 1) if item1 > list1[mid] \
        else binarySearch_rec(list1, item1, mid - 1, lower)

list1 = [4, 5, 6, 7]
print(binarySearch_rec(list1, 6, len(list1) - 1))

print(binarySearch_rec(list1, 3, len(list1) - 1))
print('')

# sorted list
# binary search needs a sorted list

list2 = [6, 7, 6, 7, 4, 5, 6, 7, 6, 7, 1, -1, 1, 2, 0, -1, 1]
list2.sort()

print(binarySearch_rec(list2, -1, len(list2) - 1))
print(binarySearch_rec(list2, 2, len(list2) - 1))

print(binarySearch_rec(list2, 3, len(list2) - 1))
print('')



# memoization
# Fibonacci and memoization

# we use: https://www.youtube.com/watch?v=Qk0zUZW-U_M

# use: http://interactivepython.org/runestone/static/pythonds/index.html#
# http://interactivepython.org/runestone/static/pythonds/Recursion/toctree.html

# Fibonacci series with recursion
def Fib_rec(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    return Fib_rec(n - 1) + Fib_rec(n - 2)

print(Fib_rec(1))
print(Fib_rec(2))

print(Fib_rec(3))
print(Fib_rec(4))

print(Fib_rec(10))
print('')

# stack overflow
# execution stack => recursion

# Fibonacci series with no recursion
def Fib(n):
    prev = 1
    last = 1

    for i in range(1, n):
        # prev = last
        # last = prev + last

        last, prev = (prev + last), last

    return last

print(Fib(1))
print(Fib(2))

print(Fib(3))
print(Fib(4))

print(Fib(10))
print('')

# website: http://interactivepython.org/runestone/static/pythonds/index.html#

# no recursion
def sumFib(n):
    prev = 1
    last = 1
    sum1 = 2

    if n == 0:
        return 1

    for i in range(1, n):
        # prev = last
        #last = prev+last

        last, prev = (prev + last), last

        sum1 += last

    return sum1

print(sumFib(3))
print(sumFib(4))

print(sumFib(10))
print('')

# recursion
def sumFib_rec(n):
    if n == 0:
        return 1
    if n == 1:
        return 2
    return 1 + sumFib_rec(n - 2) + sumFib_rec(n - 1)

print(sumFib_rec(3))
print(sumFib_rec(4))

print(sumFib_rec(10))
print('')

# tuple and list unpacking
a = [3, 4, 5, 600]

# unpack list to variables
m1, m2, m3, m4 = a
print(m4)

m1, m2 = 3, 4  #
print(m1, m2)
print('')

# the left-hand side => same elements as needed
# m1, m2 = a   error too many values to unpack

# unpack
m1, m2, *r = a
# in r => elements are stored as a list

# we use a list for "r"
m1, m2, *r = (1, 2, 3, 4, 5, 6, 7)
print(r)

m1, m2, *r = 1, 2, 3, 4, 5, 6, 7
print(r)

print('')

# use: http://interactivepython.org/runestone/static/pythonds/index.html#
# http://interactivepython.org/runestone/static/pythonds/Recursion/toctree.html



# the greatest common divisor (gcd)
# https://en.wikipedia.org/wiki/Greatest_common_divisor

# greatest common divisor (gcd)
def mkd(a, b):
    if a == b:
        return a
    # return mkd(abs(a-b), min(a,b))
    return mkd(a - b, b) if a > b else mkd(a, b - a)  # x if a>b else y

print(mkd(15, 3))
print(mkd(90, 12))

print('')

# https://www.w3resource.com/c-programming-exercises/recursion/index.php

# exit maze
# recursion problem => maze

"""
maze problem, recursion

The maze is a 2D array of integers.
The maze has 0 and 1 where 0 means empty and 1 means wall.

The input is 0,0 and the output is given.
For example, the output can be len(.)-1,len(.)-1.

The output is marked with a 3.

If I reach the output, then I have finished.
If not, then I try 4 different options => recursion.
I try 4 different options if they have 0 (and not 1).

Infinite loop => I should not pass from places I have visited.

I should not go to places I have visited before.
I mark the places I have visited with 2.

The 2 should mark the correct way in the end. I must delete the 2 if there is no way out.
"""

array2D = [[0 for i in range(6)] for j in range(6)]
print(array2D)

print(len(array2D))
print('')

for i in range(len(array2D)):
    print(array2D[i])
    print

array2D[3][4] = 1
print('')

for i in range(len(array2D)):
    print(array2D[i])
    print

# use numpy
import numpy as np

print('')

length1 = 6
array2D = np.random.randint(2, size=(length1, length1))

array2D[0][0] = 0
# array2D[len(array2D)-1][len(array2D)-1] = 0
array2D[len(array2D) - 1][len(array2D) - 1] = 3

array2D[1][0] = 0
array2D[2][0] = 0
array2D[2][1] = 0
array2D[2][2] = 0
array2D[3][2] = 0
array2D[3][3] = 0
array2D[3][4] = 0
array2D[4][4] = 0
array2D[4][5] = 0

print(array2D)
print('')

print((5, 6) == (3, 4))
print((5, 6) == (5, 6))
print('')

stack1 = [4]
print(stack1.pop())

stack1 = [4, 5, 2]
print(stack1.pop())
print('')

stack1 = []
stack2 = []

# stack1.append(0)
# stack2.append(0)

grid = array2D
maze = grid

# search the maze
def search(x, y):
    global grid

    #if grid[x][y] == 3:
    #    return True
    #elif grid[x][y] == 1:
    #    # wall
    #    return False
    #elif grid[x][y] == 2:
    #    # visited
    #    return False

    if grid[x][y] == 3:
        return True
    if grid[x][y] == 1 or grid[x][y] == 2:
        # wall
        return False
    #if grid[x][y] == 2:
    #    # visited
    #    return False

    # mark as visited
    grid[x][y] = 2

    # explore neighbors clockwise starting by the one on the right
    if (x < len(grid) - 1 and search(x + 1, y)) \
            or (y > 0 and search(x, y - 1)) \
            or (x > 0 and search(x - 1, y)) \
            or (y < len(grid) - 1 and search(x, y + 1)):
        return True

    #grid[x][y] = 0
    return False

# search(0, 0)
print(search(0, 0))
print('')

print(grid)
print('')

# base case = the end case
# we start with the base case

# recursion, maze problem
def maze_rec(array2D, endGiven, currentPosition=0, currentPosition2=0, var1=0, var2=0):
    if (currentPosition, currentPosition2) == endGiven:
        return True

    array2D[currentPosition][currentPosition2] = 2

    global stack1, stack2

    # print(currentPosition)
    # print(currentPosition2)

    stack1.append(currentPosition)
    stack2.append(currentPosition2)

    # print(array2D)

    if array2D[currentPosition + 1][currentPosition2] == 0 and 0 < currentPosition + 1 < len(array2D) - 1 and (
    currentPosition + 1, currentPosition2) != (var1, var2):
        return maze_rec(array2D, endGiven, currentPosition + 1, currentPosition2, currentPosition, currentPosition2)
    if array2D[currentPosition][currentPosition2 + 1] == 0 and 0 < currentPosition2 + 1 < len(array2D) - 1 and (
    currentPosition, currentPosition2 + 1) != (var1, var2):
        return maze_rec(array2D, endGiven, currentPosition, currentPosition2 + 1, currentPosition, currentPosition2)
    if array2D[currentPosition - 1][currentPosition2] == 0 and 0 < currentPosition - 1 < len(array2D) - 1 and (
    currentPosition - 1, currentPosition2) != (var1, var2):
        return maze_rec(array2D, endGiven, currentPosition - 1, currentPosition2, currentPosition, currentPosition2)
    if array2D[currentPosition][currentPosition2 - 1] == 0 and 0 < currentPosition2 - 1 < len(array2D) - 1 and (
    currentPosition, currentPosition2 - 1) != (var1, var2):
        return maze_rec(array2D, endGiven, currentPosition, currentPosition2 - 1, currentPosition, currentPosition2)

    print(array2D)

    print(currentPosition)
    print(currentPosition2)

    array2D[currentPosition][currentPosition2] = 0

    var1 = stack1.pop()
    var2 = stack2.pop()

    # print(stack1.pop())
    # print(stack2.pop())

    # return maze_rec(array2D, endGiven, stack1.pop(), stack2.pop(), var1, var2)
    return maze_rec(array2D, endGiven, stack1.pop(), stack2.pop(), var1, var2) if len(stack1) > 0 \
        else False

    # return maze_rec(array2D, endGiven, stack1[len(stack1)-1], stack2[len(stack2)-1])
    # return maze_rec(array2D, endGiven, stack1.pop(), stack2.pop()) if len(stack1)>0 \
    # else maze_rec(array2D, endGiven)

    # return maze_rec(array2D, endGiven, stack1.pop(), stack2.pop()) if len(stack1)>0 \
    # else False

    # array2D[currentPosition][currentPosition2] = 0

#    if array2D[currentPosition+1][currentPosition2]==2 and 0<currentPosition+1<len(array2D)-1:
#        return maze_rec(array2D, endGiven, currentPosition+1, currentPosition2)
#    if array2D[currentPosition][currentPosition2+1]==2 and 0<currentPosition2+1<len(array2D)-1:
#        return maze_rec(array2D, endGiven, currentPosition, currentPosition2+1)
#    if array2D[currentPosition-1][currentPosition2]==2 and 0<currentPosition-1<len(array2D)-1:
#        return maze_rec(array2D, endGiven, currentPosition-1, currentPosition2)
#    if array2D[currentPosition][currentPosition2-1]==2 and 0<currentPosition2-1<len(array2D)-1:
#        return maze_rec(array2D, endGiven, currentPosition, currentPosition2-1)

array2D_2 = array2D
array2D[len(array2D)-1][len(array2D)-1] = 0

# call the function maze_rec
# maze_rec(array2D, (len(array2D)-1, len(array2D)-1))
print(maze_rec(array2D, (len(array2D) - 1, len(array2D) - 1)))

print('')
print(array2D)

# recursion, maze problem
def maze_rec2(array2D, currentPosition=0, currentPosition2=0, var1=0, var2=0):
    if array2D[currentPosition][currentPosition2] == 3:
        return True

    array2D[currentPosition][currentPosition2] = 2

    global stack1, stack2

    # print(currentPosition)
    # print(currentPosition2)

    stack1.append(currentPosition)
    stack2.append(currentPosition2)

    # print(array2D)

    if array2D[currentPosition + 1][currentPosition2] == 0 and 0 < currentPosition + 1 < len(array2D) - 1 and (
    currentPosition + 1, currentPosition2) != (var1, var2):
        return maze_rec2(array2D, currentPosition + 1, currentPosition2, currentPosition, currentPosition2)
    if array2D[currentPosition][currentPosition2 + 1] == 0 and 0 < currentPosition2 + 1 < len(array2D) - 1 and (
    currentPosition, currentPosition2 + 1) != (var1, var2):
        return maze_rec2(array2D, currentPosition, currentPosition2 + 1, currentPosition, currentPosition2)
    if array2D[currentPosition - 1][currentPosition2] == 0 and 0 < currentPosition - 1 < len(array2D) - 1 and (
    currentPosition - 1, currentPosition2) != (var1, var2):
        return maze_rec2(array2D, currentPosition - 1, currentPosition2, currentPosition, currentPosition2)
    if array2D[currentPosition][currentPosition2 - 1] == 0 and 0 < currentPosition2 - 1 < len(array2D) - 1 and (
    currentPosition, currentPosition2 - 1) != (var1, var2):
        return maze_rec2(array2D, currentPosition, currentPosition2 - 1, currentPosition, currentPosition2)

    print(array2D)

    print(currentPosition)
    print(currentPosition2)

    #array2D[currentPosition][currentPosition2] = 0

    var1 = stack1.pop()
    var2 = stack2.pop()

    # return maze_rec(array2D, endGiven, stack1.pop(), stack2.pop(), var1, var2)
    return maze_rec2(array2D, stack1.pop(), stack2.pop(), var1, var2) if len(stack1) > 0 \
        else False

# call the function maze_rec
# maze_rec(array2D, (len(array2D)-1, len(array2D)-1))
print(maze_rec2(array2D_2))

print('')
print(array2D)

# use: https://www.laurentluce.com/posts/solving-mazes-using-python-simple-recursivity-and-a-search/

# we use recursion
# base case and recursive call

def solve(maze, y, x):
    if maze[y][x] == 3:
        # base case - endpoint has been found
        return True
    else:
        # search recursively in each direction from here
        return

def solveMaze_rec(maze, startRow=0, startColumn=0):
    maze[startRow][startColumn] = 2

    # base case

    #  Check for base cases:
    #  1. We have run into an obstacle, return false
    if maze[startRow][startColumn] == 1:
        return False
    #  2. We have found a square that has already been explored
    if maze[startRow][startColumn] == 2:
        return False
    # 3. Success, an outside edge not occupied by an obstacle
    if maze[startRow][startColumn] == 0:
        return True

    # Otherwise, use logical short circuiting to try each
    # direction in turn (if needed)
    found = solveMaze_rec(maze, startRow - 1, startColumn) or \
            solveMaze_rec(maze, startRow + 1, startColumn) or \
            solveMaze_rec(maze, startRow, startColumn - 1) or \
            solveMaze_rec(maze, startRow, startColumn + 1)
    if not found:
        maze[startRow][startColumn] = 0
    return found

print('')

# call the function maze_rec
# maze_rec(array2D, (len(array2D)-1, len(array2D)-1))
print(solveMaze_rec(maze))

print('')
print(maze)

# use: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# we use: http://interactivepython.org/runestone/static/pythonds/index.html#

# website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/BasicDS/toctree.html

# we use lambda expressions in Python
# use: https://docs.python.org/2/reference/expressions.html#lambda

# we use: https://docs.python.org/2/reference/expressions.html
# website: https://docs.python.org/2/reference/expressions.html#lambda



# main website: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# Compute the sum 1/2 + 3/5 + 5/8 + .... for N terms with recursion and with no recursion.

# 1/2 + 3/5 + 5/8 + 7/11 + 9/14 + ....
# sum of 1/2 + 3/5 + 5/8 + 7/11 + 9/14 + .... for N terms with recursion and with no recursion

# sum of N terms with no recursion
def functionSum(n):
    sum1 = 0
    for i in range(n):
        #sum1 += (2*n+1) / (2*n+n+2)
        sum1 += (2*i+1) / (3*i+2)

    return sum1

print('')
print(functionSum(1))
print(functionSum(2))

print(functionSum(3))
print(functionSum(4))

print(functionSum(10))
print('')

# sum of N terms with no recursion
def functionSum2(n):
    sum1 = 0
    var1 = 1
    var2 = 2
    for i in range(n):
        sum1 += var1 / var2
        var1 += 2
        var2 += 3

    return sum1

print(functionSum2(1))
print(functionSum2(2))

print(functionSum2(3))
print(functionSum2(4))

print(functionSum2(10))
print('')

# sum of N terms with recursion
def functionSum_rec(n):
    if n == 1:
        return 1/2

    #return ((2*(n-1)+1) / (2*(n-1)+(n-1)+2)) + functionSum_rec(n-1)
    return ((2*n - 1) / (3*n - 1)) + functionSum_rec(n - 1)

print(functionSum_rec(1))
print(functionSum_rec(2))

print(functionSum_rec(3))
print(functionSum_rec(4))

print(functionSum_rec(10))
print('')

# sum of N terms with recursion
def functionSum2_rec(n, var1=0, var2=0):
    if n == 1:
        return 1/2
    if (var1 == 0 and var2 == 0):
        var1 = (2*n - 1)
        var2 = (3*n - 1)
    #else:
    #    pass
    return (var1/var2) + functionSum2_rec(n-1, var1-2, var2-3)

print(functionSum2_rec(1))
print(functionSum2_rec(2))

print(functionSum2_rec(3))
print(functionSum2_rec(4))

print(functionSum2_rec(10))
print('')

# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# Find the n-term of the series: a(n) = a(n-1)*2/3 with recursion and with no recursion.

# recursion for a(n) = a(n-1)*2/3
def function1(n):
    if n == 0:
        return 1
    return (2/3) * function1(n-1)

#print('')
print(function1(1))

print(function1(2))
print(function1(3))

print(function1(9))
print(function1(10))
print('')

# no recursion for a(n) = a(n-1)*2/3
def function2(n):
    k = 1
    for i in range(1,n+1):
        k *= 2/3

    return k

#print('')
print(function2(1))

print(function2(2))
print(function2(3))

print(function2(9))
print(function2(10))
print('')

# use: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/BasicDS/toctree.html

# dry run the code
# run the code like the interpreter

# we use lambda expressions in Python
# use: https://docs.python.org/2/reference/expressions.html#lambda

# website: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# we use: https://docs.python.org/2/reference/expressions.html
# website: https://docs.python.org/2/reference/expressions.html#lambda

