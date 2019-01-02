# we use PyTorch
import torch

import numpy as np
import matplotlib.pyplot as plt

# use PyTorch and "autograd"
from torch.autograd import Variable

# define function for creating data
def makedata(numdatapoints):
    x = np.linspace(-10, 10, numdatapoints)

    #coeffs = [0.5, 5]
    # here, 5 is the bias, it is 5*x^0=5

    coeffs = [2, 0.5, 5]

    # polynomial, we evaluate a polynomial
    y = np.polyval(coeffs, x)
    # polyval is used to create a polynomial dataset

    # we now add noise, we add additive noise
    y += 2 * np.random.rand(numdatapoints)
    # rand(.) is for numbers 0 to 1

    return x,y

numdatapoints = 10

# we now create the data
inputs, labels = makedata(numdatapoints)
# we create a labeled dataset

# we plot the figure with the data
fig = plt.figure(figsize=(10, 20))

ax1 = fig.add_subplot(121)
# 121 means 1 height, 2 wide, and this is the first figure

ax1.set_xlabel("Input")
ax1.set_ylabel("Output")

#ax1.scatter(np.array(inputs), np.array(labels), s=5)
# the "s=5" is the size of the data points

# we create a scatter plot, we plot y against x
ax1.scatter(np.array(inputs), np.array(labels), s=8)

ax1.grid()

ax2 = fig.add_subplot(122)

ax2.set_title("Error vs Epoch")
ax2.grid()

line1, = ax1.plot(inputs, inputs)
# here, we do not care about the second output

# ion(.) is interactive on

# ion(.) is interactive on, we will update the graphs interactively
plt.ion()

plt.show()



def makefeatures(power):
    features = np.ones((inputs.shape[0], len(powers)))
    # len(powers) = number of columns

    for i in range(len(powers)):
        features[:,i] = (inputs**powers[i])

    print(features.shape)

    return features.T



# we define the class LinearModel
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # we create a linear layer in our model
        self.l = torch.nn.Linear(features.shape[0], 1)
        # the inputs is features.shape[0]
        # the output is 1

    def forward(self, x):
        out = self.l(x)
        return out

# the list of hyperparameters
epochs = 50

# lr is the learning rate
#lr = 0.2

#lr = 0.000003
lr = 0.000003

#powers = [1, 2]
powers = [1, 2, 3]

# we now create the features
features = makefeatures(powers)

# features.T means transpose of features

datain = Variable(torch.Tensor(features.T))
# we use transpose .T

labels = Variable(torch.Tensor(labels.T))
# we use transpose .T

# we now create our model
mymodel = LinearModel()

# we now use the MSE cost function
criterion = torch.nn.MSELoss(size_average=True)
# size_average=True means divide my m, where m is the number of training data

#criterion = torch.nn.MSELoss()

# we use stochastic gradient descent (SGD)
optimiser = torch.optim.SGD(mymodel.parameters(), lr=lr)
#lr=lr defines the learning rate, the step size of SGD



def train():
    costs = []

    for e in range(epochs):
        prediction = mymodel(datain)

        # our criteron is the MSE
        cost = criterion(prediction, labels)

        # we now use append(.), list1.append(.)
        costs.append(cost.data)

        print("Epoch", e, "Cost", cost.data[0])

        # we get our parameters out
        params = [mymodel.state_dict()[i][0] for i in mymodel.state_dict()]

        # we set our parameters equal to a list that we define
        # we use i, and when i = 1 then we get the first elements out of the dictionary

        weights = params[0]
        bias = params[1]

        optimiser.zero_grad()

        # we propagate the errors back
        cost.backward()
        # we propagate the errors back using gradients, derivatives

        optimiser.step()

        # we now define the line "line1"

        # torch.mm is matrix multiplication
        line1.set_ydata(torch.mm(weights.view(1,-1), datain.data.t()) + bias)
        # we add a bias term to the matrix-vector multiplication (i.e. to the inner product)

        fig.canvas.draw()
        ax2.plot(costs)

        plt.pause(1)

train()



# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/BasicDS/toctree.html

# we use lambda expressions in Python
# use: https://docs.python.org/2/reference/expressions.html#lambda

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
# generator objects => less memory, dynamic memory

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

# Find the n-term of the series: a(n) = a(n-1)*2/3 with recursion and with no recursion.

# Compute the sum 1/2 + 3/5 + 5/8 + .... for N terms with recursion and with no recursion.

