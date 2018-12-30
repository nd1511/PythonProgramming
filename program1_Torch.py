# main website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/Recursion/toctree.html

# use: https://docs.python.org/3/library/functions.html

# we use the Python built-in functions
# https://docs.python.org/3/library/functions.html

# we use PyTorch
import torch

# create a tensor
x = torch.Tensor([1, 2, 3])
print(x)

x = torch.Tensor([[1, 2],
                 [5, 3]])
print(x)

x = torch.Tensor([[[1, 2],
                 [5, 3]],
                [[5,3],
                 [6,7]]])

#print(x)
print(x[0][1])
# layer, row, column

# we are indexing tensors
# indexing tensor: layer row, column
print(x[0][1])

# indexing tensor: layer row, column
print(x[0][1][0])

# a variable is different from tensors

# tensors is values
# variables has values that change

from torch.autograd import Variable

# n is the number of features
n = 2

# m is the number of training points
m = 300

# m is the number of training samples, m>>n

# randn(.,.) is Gaussian with zero-mean and unit variance

# we create a matrix of depth n and breadth m
x = torch.randn(n, m)

X = Variable(x)

# we create a fake data set, we create Y

# we use: X.data[0,:], where this is the first row of the matrix X
# we use: X.data[1,:], where this is the second row of the matrix X

Y = Variable(2*X.data[0,:] + 6*X.data[1,:] + 10)

w = Variable(torch.randn(1,2), requires_grad=True)
b = Variable(torch.randn([1]), requires_grad=True)

costs = []

#import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

plt.ion()

#plt.figure()
fig = plt.figure()

ax1 = fig.add_subplot(111, projection="3d")
ax1.scatter(X[0,:].data, X[1,:].data, Y.data)

plt.show()
#matplotlib.pyplot.show()

#plt.pause(9999999999)
#plt.pause(2)

plt.pause(1)

#epochs = 500
#epochs = 100

epochs = 10

# learning rate lr
#lr = 0.5

lr = 0.1

#import numpy as np

#x1 = np.arange(-2, 10)
#x1 = np.arange(100)
x1 = np.arange(-2, 4)

#x2 = np.arange(-2, 10)
#x2 = np.arange(100)
x2 = np.arange(-2, 4)

x1, x2 = np.meshgrid(x1, x2)

for epoch in range(epochs):
    h = torch.mm(w, X) + b

    cost = torch.sum(0.5*(h-Y)**2)/m
    # to the power of 2, we use: ** 2

    cost.backward()

    w.data -= lr * w.grad.data
    b.data -= lr * b.grad.data

    w.grad.data.zero_()
    # the underscore _ means replace it with zero

    b.grad.data.zero_()
    # the underscore _ means replace it with zero

    costs.append(cost.data)

    y = b.data[0] + x1*w.data[0][0] + x2*w.data[0][1]
    s = ax1.plot_surface(x1, x2, y)

    fig.canvas.draw()

    s.remove()
    plt.pause(1)



# use: https://docs.python.org/3/library/functions.html

# we use the Python built-in functions
# https://docs.python.org/3/library/functions.html

# main website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/Recursion/toctree.html

import numpy as np

# find the number of even numbers in the list
a = [4, 5, 6, 5, 6, 7, 6, 5, 4, 5, 6]

counter0 = 0
for i in a:
    if i % 2 == 0:
        counter0 += 1

print('')
print(counter0)

def counter1(list1):
    count1 = 0
    for i in list1:
        if i % 2 == 0:
            count1 += 1

    return count1

print(counter1(a))

# find how many elements in b exist in a
b = [1, 2, 3, 5]

def counter2(list1, list2):
    count1 = 0
    for i in list2:
        for j in list1:
            if i == j:
                count1 += 1
                break

    return count1

print('')
print(counter2(a, b))

# find how many elements in b exist in a
b = [-1, 0, 2, 3, 5, 6]
print(counter2(a, b))

# find how many double entries exist in a list
def counter3(list1):
    # list2 = list(set(list1))
    list2 = set(list1)

    return len(list1) - len(list2)

print('')
print(counter3(a))

# find how many double entries exist in a list
def counter4(list1):
    count1 = 0
    for i in range(len(list1)):
        if list1[i] not in list1[:i]:
            for j in range(i + 1, len(list1)):
                if list1[i] == list1[j]:
                    count1 += 1
                    break

    return count1

# we have used: "if list1[i] not in list1[:i]:"

# "if a[i] not in a[:i]:"
# use: "if list1[i] not in list1[:i]:"

print(counter4(a))

print('')
a = [4, 5, 6, 5, 6, 7, 6, 5, 4, 5, 6]

# a[:i] is the same as a[0:i:1]
# we now use "if a[i] not in a[:i]:"

for i in range(len(a)):
    if a[i] not in a[:i]:
        print(a[i])



# use LIST COMPREHENSION
# we use: https://docs.python.org/3/library/functions.html

# find how many numbers are even using list comprehension
a = [3,4,5,4,5,6,7,8,9]

# we use list comprehensions
b = len([k for k in a if k % 2 == 0])

print('')
print(b)

b = [k for k in a if k % 2 == 0]
print(b)

b = [k**2 for k in a if k % 2 == 0]
print(b)

# use generator objects
b = (k**2 for k in a if k % 2 == 0 and k % 4 == 0)
print(list(b))

# list comprehensions in parenthesis => generator object

# we use generator objects instead of lists
# generator object are from list comprehensions in parenthesis

print('')

# we use generator objects
b = (k**3 for k in a if k % 2 == 0 and k % 4 == 0)
print(b)

# generator objects give us their values when we call them with a for loop

# generator objects give us their values only once
# a generator object gives us its values only one time when we use a for loop

# use a for loop for generator objects
for i in b:
    print (i)

# this will produce nothing due to the generator object
for i in b:
    print (i)

# create a generator object
b = (k**3 for k in a if k % 2 == 0 and k % 4 == 0)
# use dynamic memory with the generator object

print('')
# use "list(.)"
print(list(b))

# we filter a list and produce a new list
c = ((k, k+1) for k in a)
# we use tuples: we create a list of tuples

print('')
print(list(c))

# filter the list and produce a new list
c = ((k, k/2) for k in a)
print(list(c))

# "/2" is float division and "//2" is integer division
c = ((k, k//2) for k in a)
print(list(c))

# list comprehention: filter kai map
d = ('x'*k for k in a)
# "'x'*k" means repeat 'x' k times

print('')
print(list(d))

# "*i" means repeat i times
d = ('x'*i for i in a if i%2 == 1)
print(list(d))

print('')
a = [3,4,5,4,5,6,7,8,9]

# use: "if a[i] not in a[:i]"
d = ('o'*a[i] for i in range(len(a)) if a[i] not in a[:i])
print(list(d))

# filter and map
# map because we create a new list or generator object

# use "set(list1)" for no dublicates
# build-in functions: https://docs.python.org/3/library/functions.html

# list comprehentions perform filtering kai mapping
list1 = [1,2,3]
list2 = ("a", "b", "c")

# list with tuples with all combinations of 1 και 2
list3 = set((i,j) for i in list1 for j in list2)

print('')
print(list3)

list3 = set((i,j) for i in list1 for j in list2 if i % 2 == 0)
print(list3)

print('')
list2 = ("a", "", "c")

list3 = set((i,j) for i in list1 for j in list2 if i % 2 == 0 and len(j) > 0)
print(list3)

list3 = [2,4,-1,-2,2,2,8,1,8]
list4 = set(i for i in list1 for j in list3 if i==j)
print(list(list4))

list3 = [2,4,-1,-2,2,2,8,1,8]
#list4 = [i for i in list1 for j in list3 if i==j]

# use: if list1[i] not in list1[:i]
list4 = [list1[i] for i in range(len(list1)) for j in range(len(list3)) if list1[i] not in list1[:i] and \
         list3[j] not in list3[:j] and list1[i]==list3[j]]
print(list4)

# list comprehentions perform both filtering kai mapping

# use list comprehension
list1 = [-4, 4, -3, -5, 2, -1, 9]
list2 = [k**2 for k in list1]

print('')
print(list1)
print(list2)

# use list comprehention for filtering kai mapping
list2 = [(abs(k)+1)**2 for k in list1 if k%2 == 0]
print(list2)

# built-in functions: all, any
# any: return true when at least one element is true

# built-in functions: all, any
# all: return false when at least one element is false

# check if negative number exists
any(p<0 for p in a)

print('')

print(any(p<0 for p in a))
print(not all(p>=0 for p in a))

# check if x exists in the list
x = 8

print('')

#any(i==x for i in a)
print(any(i==x for i in a))

x = 100
print(any(i==x for i in a))



# we sort the list
# use either "sorted" or "sort"

# we use either "sorted" or "sort"
# "sorted" return a new list that is sorted
# "sort" sorts the list itself (not creating a new list)

print('')
a = [3, 4, -3, 2, -3, -4, 3, 31]

b = sorted(a)
a.sort()

print(a)
print(b)

# we use either "sorted" or "sort"
# "sorted": parameter reverse

a = [3, 4, -3, 2, -3, -4, 3, 31]
b = sorted(a, reverse=True)

print(b)
print('')

# "sorted": parameter key
# the parameter key is a function or a lambda expression

# we use lambda expressions

b = sorted(a, reverse=True, key=abs)
print(b)

# "sorted": parameter key is either a function or a lambda expression

# "sorted": use a function or a lambda expression
# sorting is based on the result of the function or the lambda expression

# define a function for the "sorted" parameter key
def myfunction(k):
    if k % 2 == 0:
        return k
    else:
        return k ** 2

a = [3, 4, -3, 2, -3, -4, 3, 31]
c = a

b = sorted(a, reverse=True, key=myfunction)

print('')
print(b)

# key functions can be any one-to-one function

# use: lamda parameters : return_value
# lamda expression in Python: "lamda parameters : return_value"

# use: x if x >= 0 else -x
# we can use if in lambda expressions: "x if x >= 0 else -x"

# sorting with -x
b = sorted(a, key=lambda x: -x)
# lambda expression: "lambda x: -x"

print('')
print(b)

# sorting with abs
b = sorted(a, key=abs)
print(b)

# use the previously defined function "my function"

# we use lamda expression for "my function"
b = sorted(a, key=lambda k: k if k % 2 == 0 else k ** 2)

print('')
print(b)

# we use lamda expression and if statement
b = sorted(a, key=lambda k: k if k % 4 == 0 else k ** 3)
print(b)

# we define function that has a function as a parameter
def myfunction2(list1, function1):
    for i in list1:
        function1(i)

print('')
myfunction2(a, print)  # print every value

print('')

# myfunction2(sorted(a), lambda x : print(-x+y))
myfunction2(sorted(a), lambda x: print(-x))

print('')
myfunction2(a, lambda x: print(x if x % 2 == 0 else x ** 2))
# we have use lambda expressions



# binary search requires a sorted list
# binary search uses "==", ">" and "<" than middle element
# website: http://interactivepython.org/runestone/static/pythonds/index.html#

# we use integer division: "//2"
# binary search uses "first", "last" and "middle = (first+last)//2"
# binary search: "ceiling", "floor" and both the ceiling and the floor move

# binary search needs only maximum 9 or 10 searches => very efficient, very fast, very stable

# functions use global variables
# global vs local memory variables
# def funName(localVar1, localVar2): global x,y

# we use: "if array1[i] not in array1[:i]:"
# use: "for i in array1:" and "for i in range(len(array1)):"

# use "help(list)"
array1 = [2, 5, 6, 3, 2, 2, 1, -1, 7]
array1.sort()

print('')
print(array1)

# we use straight exchange sort
# straight exchange sort = babble sort
# http://interactivepython.org/runestone/static/pythonds/index.html#

# babble sort uses swap: "x,y = y,x"
# swap: "x[i], x[i-1] = x[i-1], x[i]"
# babble sort has 2 for loops and a swap operation

# we use "if list1[i] not in list1[:i]:" when "for i in range(len(list1)):"
# use len(list1), sum(list1), set(list1) and list(set(list1))

# use list comprehension
array2 = [i**3 for i in array1 if i%2 == 0]
print(array2)

# create a generator object
# less memory, dynamic memory, use "(.)" and not "[.]"
array2 = (i**3 for i in array1 if i%2 == 0)

print('')
print(list(array2))

# generator objects exist for only one time
print(list(array2))

# perform both filtering and mapping
# we filter and map using list comprehensions
array2 = [(i, i+1) for i in array1]

print('')
print(array1)
print(array2)

array2 = [(i, i/2) for i in array1]
print(array2)

# buil-in functions: all(.), any(.), filter(.), sorted(.)

# use the buil-in function "sorted(.)"
array1 = [2, 5, 6, 3, 2, 2, 1, -1, 7]
array2 = sorted(array1, reverse = True)

print('')
print(array2)

# we now use lambda expressions
# lambda expressions are small def functions

# use lambda expressions
array2 = sorted(array1, reverse = False, key = lambda x : -x)
print(array2)

# we use lambda expressions in Python
array2 = sorted(array1, reverse = False, key = lambda input1 : input1 if input1%2 == 0 else input1+501)
print(array2)

from numpy import nan

# we use an if statement and a lambda expression
array2 = sorted(array1, reverse = False, key = lambda x : x if x%2 == 1 else nan)
print(array2)

