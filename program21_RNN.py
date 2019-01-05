# we use: http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf
# we use: F. Chollet, Deep learning with Python

# RNN
# an RNN is a for loop that reuses quantities computed during the previous iteration of the loop

# RNNs, LSTM-RNNs and GRU-RNNs
# LSTM-RNNs and GRU-RNNs are better than RNNs

# LSTM-RNNs and GRU-RNNs are better than vanilla RNNs
# LSTM-RNNs are used in (https://research.google.com/pubs/archive/44312.pdf)
# GRU-RNNs are used in (https://pdfs.semanticscholar.org/702e/aa99bcb366d08d7f450ed7e354f9f6920b23.pdf)

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

# use for loop
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)

    # we define the output
    #output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    # we use tanh

    successive_outputs.append(output_t)

    state_t = output_t

    final_output_sequence = np.concatenate(successive_outputs, axis=0)



# we use SimpleRNN
from keras.layers import SimpleRNN

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

# plot the training and validation loss
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()

plt.show()



# use: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# we use: http://interactivepython.org/runestone/static/pythonds/index.html#

# website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/BasicDS/toctree.html

# we use lambda expressions in Python
# use: https://docs.python.org/2/reference/expressions.html#lambda

# we use: https://docs.python.org/2/reference/expressions.html
# website: https://docs.python.org/2/reference/expressions.html#lambda

# recursion, Fibonacci series
# recursion and memoization to solve stack overflow problems

# we use: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# Fibonacci series with recursion
def Fib_rec(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    return Fib_rec(n - 1) + Fib_rec(n - 2)

print(Fib_rec(1))
print(Fib_rec(2))

print(Fib_rec(10))
print('')

# Fibonacci series with no recursion
def Fib(n):
    prev = 1
    last = 1

    for i in range(1, n):
        # prev = last
        last, prev = (prev + last), last

    return last

print(Fib(1))
print(Fib(2))

print(Fib(10))
print('')

# sum of N terms from Fibonacci
def sumFib(n):
    prev = 1
    last = 1
    sum1 = 2

    if n == 0:
        return 1

    for i in range(1, n):
        # prev = last
        last, prev = (prev + last), last

        sum1 += last

    return sum1

print(sumFib(3))
print(sumFib(4))

print(sumFib(10))
print('')

# sum of N terms from Fibonacci
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

# print(Fib(1))
# print(Fib(2))
# print(Fib(10))

# memoization
# recursion and memoization

# Fibonacci series and memoization

# tuple and list unpacking
a = [3, 4, 5, 600]

# unpack list to variables
m1, m2, m3, m4 = a
print(m4)

m1, m2 = 3, 4  #same as: m1, m2 = (3, 4)
print(m1, m2)
print('')

# the left-hand side => same elements as needed
# m1, m2 = a => error too many values to unpack

# unpack
m1, m2, *r = a
# in r => elements are stored as a list

# we use a list for "r"
m1, m2, *r = (1, 2, 3, 4, 5, 6, 7)
print(r)

# use unpacking
m1, m2, *r = 1, 2, 3, 4, 5, 6, 7
print(r)

# var1 = 4 and list1 = [3,2,1]
var1, *list1 = 4,3,2,1
# we use tuples and unpacking

print(list1)
print('')

# to terminate a while loop: use "break" or "pass"

# use: "and", "or"
# and, or, in, not in, !=, ==

# De Morgan's laws
# use: https://en.wikipedia.org/wiki/De_Morgan%27s_laws

# use: while not found and lower<=upper
# we use: while not found and not lower>upper
# we now use: while not (found or lower>upper)

# we use De Morgan's laws
# use: while not (found or lower>upper)

# https://en.wikipedia.org/wiki/De_Morgan%27s_laws
# use: https://www.tutorialspoint.com/computer_logical_organization/demorgan_theroems.htm



# use: http://interactivepython.org/runestone/static/pythonds/SortSearch/TheBinarySearch.html
# we use: http://interactivepython.org/runestone/static/pythonds/index.html#

# binary search => sorted list
# binary search is stable => 9 searches to find item

# binary search needs a sorted list
def binarySearch(list1, item1):
    upper = len(list1) - 1
    lower = 0

    found = False

    while not found and upper >= lower:
        mid = (upper + lower) // 2

        #print(upper)
        #print(lower)

        if list1[mid] == item1:
            found = True
        elif list1[mid] < item1:
            lower = mid + 1
        else:
            upper = mid - 1

        # if upper < lower:
        #    break

    return found

list1 = [4, 5, 6, 7]
print(binarySearch(list1, 6))

list1 = [4, 5, 6, 7, 5, 2, 3, 1, -1, -3, -2, 0]
list1.sort()

print(binarySearch(list1, 6))
print(binarySearch(list1, -4))
print('')

# the greatest common divisor (gcd)
# use: https://en.wikipedia.org/wiki/Greatest_common_divisor

# greatest common divisor
def gcd(a, b):
    if a == b:
        return a

    # return mkd(abs(a-b), min(a,b))
    return gcd(a - b, b) if a > b else gcd(a, b - a)
    # return x if a>b else y

# use: return x if a>b else y
# we use: x if a>b else y

print(gcd(15, 3))
print(gcd(12, 18))
print(gcd(90, 12))

# https://www.w3resource.com/c-programming-exercises/recursion/index.php

# recursion programming exercises
# use: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# we use: https://www.geeksforgeeks.org/recursion-practice-problems-solutions/

# http://interactivepython.org/courselib/static/thinkcspy/Recursion/ProgrammingExercises.html
# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# recursion coding exercises
# Python recursion exercises
# recursion programming exercises



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

print('')
print(function1(1))

print(function1(2))
print(function1(3))

print(function1(9))
print('')

# no recursion for a(n) = a(n-1)*2/3
def function2(n):
    k = 1
    for i in range(1,n+1):
        k *= 2/3

    return k

print('')
print(function2(1))

print(function2(2))
print(function2(3))

print(function2(9))
print('')

# use: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/BasicDS/toctree.html



# we use lambda expressions in Python
# use: https://docs.python.org/2/reference/expressions.html#lambda

# website: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# we use: https://docs.python.org/2/reference/expressions.html
# website: https://docs.python.org/2/reference/expressions.html#lambda

import numpy as np

# we use Python's build-in functions
# use: https://docs.python.org/3/library/functions.html

# we use *args and **kwargs
# https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/

# use one-line code
# write as few lines of code as possible

# use comprehensions
a = [i for i in range(2, 100 + 1, 2)]
print(a)

# we use list comprehensions
a = [i for i in range(1, 101) if i % 2 == 0]
print(a)

# create a generator object, use "(.)"
a = (i for i in range(1, 101) if i % 2 == 0)
# the generator object can be used only once

# the generator object can be used one time only
print(list(a))
print('')

# positional arguments => position matters
# we can call function1 using "function1(y=1, x=2)"

# function with positional arguments x, y
def function1(x, y):
    return x - y

# positional arguments: the position matters
print(function1(3, 5))

# named arguments, no matter the order
print(function1(y=3, x=5))

# both positional arguments and named arguments
print(function1(4, y=7))
# in functions, position can matter and can not matter

# positional arguments for function
# positional parameters, function inputs, arguments

print('')
print(max(2,6,9,3))
print(sum([2,6,9,3]))

# functions can have default values

# define a function with default values
def func2(x, y=9, z=1):
    # the default value is for z
    return (x + y) * z
    # If we do not give a value for z, then z=1=(default value)

# we can have default values in functions
# default values go to the end of the arguments

# use: (1) default values, (2) *args, (3) **kwargs
# we use default values, one asterisk (i.e. *) and two asterisks (i.e. **)

# we now use *args and **kwargs
# use: https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/

# default arguments can be only at the end, even more than one
g = func2(2, 5, 7)
print(g)

print('')
for i in range(5):
    print(i, "-", i ** 2)

# use *args at the end
# we use un-named arguments *args

# (1) *args at the end of the arguments in a function
# (2) default values at the end of the arguments in a function

# *args must be in the end of the arguments
def apodosi(*apodoseis):
    k = 1
    for i in apodoseis:
        k *= i

    return k

# use: (1) *args, and (2) **kwargs
# "**kwargs" is a dictionary dict

# we use keys and values
# "**kwargs" is a dictionary and has keys and values

# **kwargs must be at the end and hence after *args
def apodosi(*apodoseis, **kwargs):
    # we use the "max" key in the dictionary
    if "max" in kwargs:
        n = kwargs["max"]
    else:
        n = len(apodoseis)

    k = 1
    for i in range(n):
        k *= apodoseis[i]

    return k

# **kwargs must be at the end and hence after *args
def apodosi2(*apodoseis, **kwargs):
    # we use the "max" key in the dictionary
    if "max" in kwargs:
        # we use min(., len(apodoseis))
        n = min(kwargs["max"], len(apodoseis))
    else:
        n = len(apodoseis)

    k = 1
    for i in range(n):
        k *= apodoseis[i]

    return k

print('')
print(apodosi(1.11, 1.22, 1.31))

print(apodosi2(1.11, 1.22, 1.31))
print('')

# use: "*m" amd "myFunction(*m)"
m = [2.3, 1.4, 1.8, 1.5, 2.4]

# we use: "*m" amd "myFunction(*m)"

# when we have a list m, then we use "*m" to get its elements
print(apodosi(*m, max=3))
print(apodosi2(*m, max=3))

# use *list1 to break the list
print(apodosi2(*m, max=13))
# the function does not work if we do not use "*"

# use *args and **kwargs in functions
# website: https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/

# use: https://www.geeksforgeeks.org/args-kwargs-python/



# convert to binary
# convert the number n to binary
n = 14

# we use the stack data structure
# define a list that will be used as a stack
stack1 = []

# stack: the last item that enters the stack is the first item out
# the stack data structure is Last In First Out (LIFO)
# the queue data structure is First In First Out (FIFO)

print('')

# every program uses an execution stack
# the execution stack in Python is short

# Every program has a stack that contains the parameters and the local variables of the functions
# that have been called. The stack is LIFO. The last parameter of a function gets out first, i.e. LIFO,
# when many funnctions have been called in a recursion.

# recursion problems
# recursion and memoization
# Fibonacci series and memoization

# the stack overflow error
# stack overflow: when recursion, when the execution stack is full

# we use a while loop
while n != 0:
    # d is the last digit
    d = n % 2

    # print(d)
    stack1.insert(0, d)

    # we remove the last digit
    n = n // 2

# print the elements
for i in stack1:
    print(i, end="")
print()

def toBinary(n):
    if n == 0:
        return
    toBinary(n // 2)
    print(n % 2, end='')

toBinary(14)
print()

toBinary(14)
print()

# d is the last digit
# d = n % 2
# stack1.insert(0, d)

# we remove the last digit
#n = n // 2

# we use base 8
def toOctal(n):
    if n == 0:
        return
    toOctal(n // 8)
    print(n % 8, end='')

# use base 10
def toDecimal(n):
    if n == 0:
        return
    toDecimal(n // 10)
    print(n % 10, end='')

# 453%10 = 3 = last digit
# 453//10 = 45 = remove last digit

# x%10 = last digit
# x//10 = remove last digit

# we use base 3
def toTernary(n):
    if n == 0:
        return
    toTernary(n // 3)
    print(n % 3, end='')



# sum of N numbers
def sumToN(N):
    sum = 0
    for i in range(1, N + 1):
        sum += i
    return sum

# recursion, sum of N numbers
def sumToN_rec(N):
    #print(N)
    if N == 1:
        return 1
    # return 1 + sumToN_rec(N-1)
    return N + sumToN_rec(N - 1)

print('')
print(sumToN_rec(4))

#print(sumToN_rec(40000))
print(sumToN_rec(40))

# recursion problems
# coding recursion exercises
# programming recursion exercises

# recursion and memoization
# write code with and without recursion

# use one-line code
# lambda expressions => one line only
# comprehensions, list comprehensions => one line only



# use comprehensions: lists or generator objects

# comprehensions with "(.)" => generator objects
# generator objects are created for one time only

# positional arguments
# define functions and call them with positional arguments
# positional arguments or non-positional arguments, default values

# default values go at the end, *args goes at the end
# use *args and **kwargs, **kwargs goes at the end

# use function1(*list1), use "*list1"
# we use "*list1" to break the list to its elements

# dictionary: keys and values
# dictionaries have keys and values

# we use *args and ** kwargs
# website: https://www.geeksforgeeks.org/args-kwargs-python/

# **kwargs => named arguments, dictionary

# dictionary has keys and values
# we use keys as an index to acccess the values
# "if "max" in dict1:": "max" is a key and not a value

# stack data structure => LIFO
# LIFO, last in first out, stack, execution stack
# recursion, memoization, execution stack, stack overflow

# limited stack, limited short execution stack

# recursion, Fibonacci series => stack overflow
# memoization, we use lookup table, memoization to store values

# Find the n-term of the series: a(n) = a(n-1)*2/3 with recursion and with no recursion.

# recursion for a(n) = a(n-1)*2/3
def function1(n):
    if n == 0:
        return 1
    return (2/3) * function1(n-1)

print('')
print(function1(1))

print(function1(2))
print(function1(3))

print(function1(9))
print('')

# no recursion for a(n) = a(n-1)*2/3
def function2(n):
    k = 1
    for i in range(1,n+1):
        k *= 2/3

    return k

print('')
print(function2(1))

print(function2(2))
print(function2(3))

print(function2(9))
print('')

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

print(functionSum(1))
print(functionSum(2))

print(functionSum(3))
print(functionSum(4))

print(functionSum(10))
print('')

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

