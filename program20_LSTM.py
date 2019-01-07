# we use Keras
# use: http://crcv.ucf.edu/courses/CAP6412/Spring2018/KerasTutorial.pdf

# http://zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf

# website: http://crcv.ucf.edu/courses/CAP6412/Spring2018/KerasTutorial.pdf
# use: http://zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf

# AI: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we use Keras
import keras

# use numpy
import numpy as np

# download file
path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

text = open(path).read().lower()
print('Corpus length:', len(text))

maxlen = 60
step = 3

sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])

    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))
print('Unique characters:', len(chars))

char_indices = dict((char, chars.index(char)) for char in chars)
print('Vectorization...')

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)

y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1

    y[i, char_indices[next_chars[i]]] = 1



# use Keras
from keras import layers

# SOS: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# website: http://crcv.ucf.edu/courses/CAP6412/Spring2018/KerasTutorial.pdf
# use: http://zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf

# we use Keras
# use: http://crcv.ucf.edu/courses/CAP6412/Spring2018/KerasTutorial.pdf

# http://zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf

model = keras.models.Sequential()

# we use LSTM
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))

model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)

#print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print(model.summary())



# sample the next character given the model’s predictions

# define sample
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

import random
import sys

for epoch in range(1, 60):
    print('epoch', epoch)

    model.fit(x, y, batch_size=128, epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    generated_text = text[start_index: start_index + maxlen]

    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)

        sys.stdout.write(generated_text)

        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))

            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]

            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)



# website: https://docs.quandl.com

# we use Quandl
import quandl

# use: https://www.quandl.com/databases/SF1/documentation?anchor=
# we use: https://docs.quandl.com

# https://www.quandl.com/databases/SF1/documentation?anchor=dimensions

# https://www.quandl.com/databases/SF1/documentation?anchor=methodology
# use: https://www.quandl.com/databases/SF1/documentation?anchor=dimensions

# There are 3 time dimensions:
# Annual (Y): Annual observations of one year duration
# Trailing Twelve Months (T): Quarterly observations of one year duration
# Quarterly (Q): Quarterly observations

# website: https://www.quandl.com/databases/SF1/documentation?anchor=dimensions

# we use: http://interactivepython.org/runestone/static/pythonds/index.html#
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

"""
Graphs
We use dict for graphs.
"""

# use vertex
# Graphs => vertex and graph[vertex]

# DFS = depth first search
def dfs(graph, start):
    visited, stack = set(), [start]

    while stack:
        vertex = stack.pop()

        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)

    return visited

# DFS => depth first search
def dfs_paths(graph, start, goal):
    stack = [(start, [start])]

    while stack:
        (vertex, path) = stack.pop()

        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

# BFS = breadth first search
def bfs(graph, start):
    '''
    help bfs: breadth first search
    '''
    visited, queue = set(), [start]

    while queue:
        vertex = queue.pop(0)

        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)

    return visited

# BFS => breadth first search
def bfs_paths(graph, start, goal):
    queue = [(start, [start])]

    while queue:
        (vertex, path) = queue.pop(0)

        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

graph1 = {'A': set(['B', 'C']),
          'B': set(['A', 'D', 'E']),
          'C': set(['A', 'F']),
          'D': set(['D']),
          'E': set(['B', 'F']),
          'F': set(['C', 'E'])}

print(dfs(graph1, 'A'))
print(list(dfs_paths(graph1, 'C', 'F')))

print(bfs(graph1, 'A'))
print(list(bfs_paths(graph1, 'C', 'F')))
print(list(bfs_paths(graph1, 'A', 'F')))



"""
dict = HashMap
"""

# use dictionary dict
def function1(dict1, key1, value1):
    if key1 not in dict1:
        dict1[key1] = 0

    dict1[key1] += value1
    return dict1

# use dictionary dict
def function2(dict1, key1, value1):
    dict1[key1] = dict1.get(key1, 0) + value1
    return dict1

d1 = {
    'milk': 3.67,
    'butter': 1.95,
    'bread': 1.67,
    'cheese': 4.67
}
print(d1)

d1['butter'] = 2.35
print(d1)

d2 = {
    1: 3.67,
    2: 1.95,
    3: 1.67,
    4: 4.67
}
print(d2)

d2[2] = 2.35
print(d2)

d3 = dict([('milk', 3.76), ('butter', 1.95), ('bread', 1.67), ('cheese', 4.67)])
print(d3)

del d3['butter']
print(d3)

print('len of dictionary d3 = {} se sxesi {} i to {} '.format(len(d3), d1, d2))
print(d3.keys())

print(d3.values())
print(d3.items())

# traverse a dict
for food in d1:
    print('{} costs {}'.format(food, d1[food]))

d1 = function1(d1, 'whine', 4.15)
d1 = function1(d1, 'milk', 1)
print(d1)

d1 = function2(d1, 'whine2', 3.15)
d1 = function2(d1, 'milk', 1)
print(d1)

# dict comprehension
d4 = {k: v for k, v in enumerate('Good Year John')}
print(d4)

# dict with all letters in "Good Year John"
# without the letters in "John"

d5 = {k: v for k, v in enumerate("Good Year John") if v not in "John"}
print(d5)



"""
Sets
"""

s = {'a','b','a','c','d'}
print(s)

s2 = set("Good Year John")
print(s2)

a = set('12345678a')
b = set('1234b')
print('A = ',a)
print('B = ',b)

a.add('9')
b.remove('4')
print('A = ',a)
print('B = ',b)

print('A - B = ',a-b) #difference
print('A | B = ',a|b) #Union

print('A & B = ',a&b) #intersection
print('A ^ B = ',a^b) #symmetric difference

# Venn diagram
# use Venn diagram for sets

"""
list comprehension
result = [transform iteration filter]

filter and iteration
iteration and filter
"""

lista = [1,2,3,4,5,6,7,8,9]
#print([i*2 for i in lista])

lista.pop()
print(lista)

# use list comprehension
list1 = [i for i in range(1,101) if i%3 == 0 and i%7 == 0]
print(list1)

# 2D array 6x6 with zeros
array2D = [[0 for i in range(0,6)] for j in range(0,6)]
print(array2D)

array2D[0][0] = 1
print(array2D)

# 3D array 6x6 with zeros
array3D = [[[0 for i in range(6)] for j in range(6)] for k in range(6)]

print(array3D)
print(array3D[0][0][0])

# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# binary search => for sorted list
def binarySearch(list1, item1):
    upper = len(list1) - 1
    lower = 0

    found = False

    while not found and upper >= lower:
        mid = (upper + lower) // 2

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

# we use: http://interactivepython.org/runestone/static/pythonds/index.html#
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf



# memoization
# Fibonacci series and memoization

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

# sum of Fibonacci terms
# website: http://interactivepython.org/runestone/static/pythonds/index.html#

# sum of Fibonacci terms with no recursion
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

# sum of Fibonacci terms with recursion
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



# https://www.w3resource.com/c-programming-exercises/recursion/index.php

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

# we use Keras
# use: http://crcv.ucf.edu/courses/CAP6412/Spring2018/KerasTutorial.pdf

# http://zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf

# website: http://crcv.ucf.edu/courses/CAP6412/Spring2018/KerasTutorial.pdf
# use: http://zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf

# AI: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# https://www.w3resource.com/c-programming-exercises/recursion/index.php



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

# we use both *args and **kwargs
# website: https://www.geeksforgeeks.org/args-kwargs-python/

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
        sum1 += (2*i+1) / (2*i+i+2)

    return sum1

print(functionSum(1))
print(functionSum(2))

print(functionSum(3))
print(functionSum(4))

print(functionSum(10))
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

