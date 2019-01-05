# PCA, Principal Component Analysis
# perform dimensionality reduction

# PCA: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# unsupervised learning: clustering and dimensionality reduction
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we want to find the direction in which the variance is the largest
# we find the maximum-variance direction

# We use matrices and transformations
# we can use 2 dimensions to visualize our transformation

# variance, sigma_squred = (1/M) \times \sum (x - \mu)^2
# where M data points

# covariance matrix, how does one feature vary as another feature varies
# we find the eigenvectors of the covariance matrix

# the diagonals of the covariance matrix equal to 1

# (1) compute the covariance matrix
# (2) eigenvalues and eigenvectors of the covariance matrix
# (3) keep the eigenvectors with the largest eigenvalues



# we now implement PCA
# we do dimensionality reduction

# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we import libraries
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# we use sklearn to load datasets
from sklearn import datasets

# we load the Iris dataset from sklearn
data = datasets.load_iris()
#print(data)

# we now define the variables X and Y, our data
X = data.data
Y = data.target

m = X.shape[0]

# function to normalise data
def normalise(x):
    x_std = x - np.mean(x, axis=0)

    # we use np.std(.) to compute the standard deviation
    x_std = np.divide(x_std, np.std(x_std, axis=0))

    return x_std

# decompose data
def decompose(x):
    cov = np.matmul(x.T, x)

    print('\n Covariance matrix')
    print(cov)

    # eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eig(cov)

    print('\n Eigenvectors')
    print(eig_vecs)

    print('\n Eigenvalues')
    print(eig_vals)

    return eig_vals, eig_vecs, cov

# use eigenvectors and eigenvalues

# we now find which eigenvectors are important
def whicheigs(eig_vals):
    total = sum(eig_vals)

    # we use descending order, we use "sorted(eig_vals, reverse=True)"

    # we define the variance percentage
    var_percent = [(i/total)*100 for i in sorted(eig_vals, reverse=True)]

    cum_var_percent = np.cumsum(var_percent)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Variance along different principal components')

    ax.grid()

    plt.xlabel('Principal component')
    plt.ylabel('Percentage total variance accounted for')

    ax.plot(cum_var_percent, '-ro')

    ax.bar(range(len(eig_vals)), var_percent)

    plt.xticks(np.arange(len(eig_vals)), ('PC{}'.format(i) for i in range(len(eig_vals))))

    plt.show()

# we now call the functions
X_std = normalise(X)

eig_vals, eig_vecs, cov = decompose(X_std)

whicheigs(eig_vals)



# reduce the dimensions
def reduce(x, eig_vecs, dims):
    W = eig_vecs[:, :dims]

    print('\n Dimension reducing matrix')
    print(W)

    return np.matmul(x,W), W

colour_dict = {0:'r', 1:'g', 2:'b'}

colour_list = [colour_dict[i] for i in list(Y)]

def plotreduced(x):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # "x[:,0]" is the first principal component
    #ax.scatter(x[:,0], x[:,1], x[:,2])

    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colour_list)

    plt.grid()
    plt.show()

# we now call the functions
X_std = normalise(X)

eig_vals, eig_vecs, cov = decompose(X_std)

whicheigs(eig_vals)

#X_reduced, transform = reduce(X_std, eig_vecs, 3)

dim = 3
#dim = 1
X_reduced, transform = reduce(X_std, eig_vecs, dim)

# we plot the graph with the reduced data
plotreduced(X_reduced)

# define epochs
epochs = 10

def k_means(x, y, centroids=3):

    # we use 3 dimensions
    positions = 2*np.random.rand(centroids, 3) - 1

    m = x.shape[0]

    # for each epoch
    for i in range(epochs):
        assignments = np.zeros(m)

        # for each point in the data
        for datapoint in range(m):

            # compute the difference between centroid and datapoint
            difference = X_reduced[datapoint] - positions

            # we use the Euclidean distance
            norms = np.linalg.norm(difference, 2, axis=1)

            assignment = np.argmin(norms)

            assignments[datapoint] = assignment

        # for each centroid
        for c in range(centroids):
            positions[c] = np.mean(x[assignments == c])

    print('\n Assignments')
    print(assignments)

    print('\n Labels')
    print(Y)

    # print the positions of the centroids
    print(positions)

    # return the positions
    return positions

# we do K-means
k_means(X_reduced, Y, 3)



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

# def sumFib_rec2(n):
#    prev = 1
#    last = 1
#    sum1 = 2
#
#    if n == 0:
#        return 1
#
#    for i in range(1, n):
#        #prev = last
#        last, prev = (prev + last), last
#
#        sum1 += last
#
#    return sum1

# print(Fib(1))
# print(Fib(2))
# print(Fib(10))

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

