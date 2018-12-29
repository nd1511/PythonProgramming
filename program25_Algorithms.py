# Vazirani Algorithms, Book
# use: http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf

# https://github.com/haseebr/competitive-programming/blob/master/Materials/Algorithms%20by%20Vazirani%20and%20Dasgupta.pdf
# we now use: http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf

# Cracking the Coding Interview, Book
# use: https://github.com/jwang5675/ctci/blob/master/Cracking%20the%20Coding%20Interview%206th%20Edition.pdf

# we use: http://faculty.washington.edu/pisan/cpp/readings/McDowell_ch11_Advanced%20Topics.pdf

# Data Science GitHub: https://www.analyticsvidhya.com/blog/2018/12/best-data-science-machine-learning-projects-github
# generative adversarial networks (GANs): https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/

# use: https://www.analyticsvidhya.com/blog/2018/12/best-data-science-machine-learning-projects-github

# use: https://sigmoidal.io/machine-learning-for-trading/
# we use: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2756331

# we use HackerRank
# use: https://www.hackerrank.com/nd1511
# nd1511, Nikolaos Dionelis, HackerRank

# we use: https://docs.python.org/3.6/library/index.html
# use: https://docs.python.org/3.6/library/stdtypes.html#sequence-types-list-tuple-range

# we define binary search
# binary search needs a sorted list

# binary search is very fast: finite searches even for a very long list

# define the function for binary search
def binarysearch(array, key):
    found = False

    # define the first, define the floor
    first = 0

    # define the last, define the ceiling
    last = len(array) - 1

    # binary search => define the first and last
    # binary search => we will change the first and last

    # in binary search, the floor moves
    # in binary search, the ceiling moves

    while first <= last and not found:
        # integer division
        mid = (first + last) // 2

        # use integer division
        # we use "//2" and not "/2"

        if array[mid] == key:
            found = True

        elif array[mid] < key:
            first = mid + 1

        else:
            last = mid - 1

    return found

# we define sequential search
# sequal search is brute force

# define function for serial search
def serialsearch(array, key):
    found = False

    for i in array:
        if i == key:
            found = True

    return found

# the serial search has complexity = O(N)
# O(N) is bad complexity: when we double the data, we double the time

# main program

# use time.clock()
import time

# l = [2,28,35,17,'a',6]

l = []
for i in range(-3000000, 3000000):
    l.append(i)

m = []
for i in range(3000):
    m.append(i)

print(time.clock(), 'Binary Search Start')
for i in m:
    binarysearch(m, i)
print(time.clock(), 'Binary Search End')

# print(time.clock, 'Binary Search 2 Start')
# for i in l:
#    binarysearch(l, i)
# print(time.clock, 'Binary Search 2 End')

print(time.clock(), 'Serial Search Start')
for i in m:
    serialsearch(m, i)
print(time.clock(), 'Serial Search End')

# for i in l:
#    serialsearch(l, i)

# l[l.index('a')]='1821'
# l[l.index('a')]=1821
#
# for i in range(0, len(l), 1):
#    if (l[i] == '1821'):
#        print('found')
#
# for i in l:
#    if i == '1821':
#        print('found')

# print( binarysearch(l, '1821') )
print(binarysearch(l, 1821))



# global x,y
# use global memory variables

# global variables => change the memory in the main program
# global variables => use less memory

# define the function mysum()
def mysum(a):
    # define x,y as global variables
    global x, y

    # global vs local memory variables
    # local: mysum(x, y, a)

    x = a * x
    y = y + a

    print('x in fuction = ', x)
    print('y in function = ', y)

    # return statement
    return x + y

# local memory variables: a
# functions have global variables, local variables and a return statement

# main program

# input from the user
x = int(input("User, give me the first number: "))
y = int(input("User, give me the second number: "))

print('x before function = ', x)
print('y before function = ', y)

a = 10
print(mysum(a))

print('x after function = ', x)
print('y after function = ', y)



# use: https://www.springboard.com/blog/data-science-interview-questions/#programming

# we use: https://www.springboard.com/blog/python-interview-questions/
# use: https://www.springboard.com/blog/data-science-interview-questions/

import numpy as np

# use pandas for .csv files
import pandas as pd

#help(list)
# use "help(list)" to read description

# help(file) and help(list)
# use: https://docs.python.org/3.6/library/index.html

# https://docs.python.org/3.6/library/stdtypes.html#sequence-types-list-tuple-range

list1 = []

# from -100 to 100
#for i in range(-100, 100+1, 2):
#    list1.append(i)

# from -90 to 100
for i in range(-90, 100+1, 2):
    list1.append(i)

print(list1)
print("")

# the first: print(list1[0])
# the last: print(list1[-1])

print(list1[0])
print(list1[-1])
print("")

print(list1[::1])
print("")

print(list1[::2])
print("")

# reverse the list
print(list1[::-1])
print("")

# reverse the list
#list1.reverse()

# reverse the list
#print([list1[-1:0:-1], list1[0]])
#print([list1[-1:0:-1], [list1[0]]])
print(list1[-1:0:-1] + [list1[0]])
print("")

#help(list)
print(list1[::-2])
print("")

# Kaggle competitions
# https://www.kaggle.com/nd1511
# Hackathons and Kaggle

list2 = ['Nikolaos', 23]

print(list2[0])
print(list2[1])
print("")

#print(list2[2])
#print("")

list2 = [23, 'hello', 145]

print(list2[0])
print(list2[1])
print(list2[2])
print("")

#print(list2[3])
#print("")

print(list2[1][0])
print(list2[1][1])
print("")

# we use: in range(1, n+1, 1)
# use: in range(start, end+1, step size)

# we use for loop
# for i in range(start, end+1, step size)

# if statements
# if else statements
# for loops

# factorial, !
# 0! = 1, 1! = 1

n = int(input("Give a Number: "))

p = 1

# use: range(1, n+1, 1)
for i in range(1, n+1, 1):
    p = p * i

#print(p)
print(n, "! = ", p)

# dry run the code
# run the code like the interpreter

# range(startValue, endValue+1, stepSize)
# range(1,x+1,1): until x only

# recursion, we use return
# function, def, return

# the base case ends the recursion

# functions: use def and return
def factorial(x):
    p = 1

    # use: range(StartValue, EndValue+1, StepSize)
    for i in range(1, x+1, 1):
        p = p * i

    print(x, "! = ", p)

factorial(n)

# recursive implementations are neater
# recursive implementations are more understandable

# recursive implementations take more memory and are slower

"""
factorial
n! = 1*2*3*..*n, where 1! = 1, 0! = 1
3! = 1 * 2 * 3 = 6
"""
print("")

# define function for factorial
def factorial(x):
    #print("I am in the factorial function.")
    p = 1
    for i in range(1,x+1,1):
        p = p*i

    print(n,"! = ",p)

# Recursive: The function calls itself.

# define function for recursive factorial
def recfactorial(x):
    #print("I am in the recfactorial function.")
    if x == 1:
       return 1
    else:
       return x * recfactorial(x-1)

# main program
#n = int(input("Give me a number "))

factorial(n)

print(n,"! = ", recfactorial(n), ' with recursive')
print("")

print("The program is back in the main program.")
print("")

# recursion has a base case, which is the simplest case
# recursion has a recursive call, which is the recursion step

# recursive implementations take more memory and are slower

# if we dry run the recursive implementation,
# then we go downwards searching and upwards completing
# if n=3, then:
# 3*recfactorial(2)         =6
# 2*recfactorial(1)         =2
# 1                         =1

# if we dry run the recursive implementation,
# then we go downwards searching and upwards completing
# if n=5, then:
# 5*recfactorial(4)         =120
# 4*recfactorial(3)         =24
# 3*recfactorial(2)         =6
# 2*recfactorial(1)         =2
# 1                         =1

# recursion for factorial
# factorial: n! = n * (n-1)!, if n > 1 and f(1) = 1
# we use range(StartValue, EndValue+1, StepSize)

# https://www.codecademy.com/nikolaos.dionelis
# Code Academy, Nikolaos Dionelis

# https://www.codecademy.com/nikolaos.dionelis

# Hacker Rank, Nikolaos Dionelis
# https://www.hackerrank.com/nd1511



# recursion produces easy to read code
# recursion => base case and recursion step

# the Fibonacci series => recursion
# recursion-based problems and the Fibonacci series

# memoization
# use both memoization and recursion

# Fibonacci with memoization: what memoization structure, local vs global

# Fibonacci with memoization
def fibHelper( n, memoarray ) :
    if( n in memoarray ):
        # code todo
        return
def fib( n ):
    return fibHelpber( n, [] )

# we use memoization and: https://stackoverflow.com/questions/7875380/recursive-fibonacci-memoization
# https://medium.com/@nothingisfunny/memoization-improving-recursive-solution-for-fibonacci-sequence-problem-c02dab7a74e5

# fib(100) => fib(99), fib(98)
# fib(99) => fib(98), fib(97)
# fib(98) => fib(97), fib(96)
# fib(97) => fib(96), fib(95)
# (...)

# Recursion depth
# Tree => use levels, 1 number in first level, 2 numbers in second level, 4, 8, ...

# Execution stack, stack data structure
# Recursion => Python has small execution stack

# stack data structure
# stack and queue data structures

# define the function for the Fibonacci series
def fib(n) :
    if  n == 0 :
        return 1
    elif n == 1:
        return 1
    else:
        #Return a(n - 1) + a(n - 2)
        return fib(n-1) + fib(n-2)

# fib(98) => fib (97, 96)
# fib(99) => fib (98, 97)

# 100
# 99 98
# 98 97 97 96
# 97 96 96 95 96 95 95 94
# ...

# Vazirani's Book - Algorithms
# we use: http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf

# file: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# functions have global or local variables
# global variables change the memory in the main program

# def myFunction(): global x,y
# use "global x,y" for less memory

list1 = ['a', 5, 4, -1, 'g', 'a', 'a', 3]
# use: "help(list)"
print(list1.count('a'))

print('')

# traverse the list
#for i in list1:
for i in range(len(list1)):
    # we use: "if list1[i] not in list1[:i]:"
    if list1[i] not in list1[:i]:
        print(list1[i])

# O(n) complexity is bad because double the elements means double the time

# binary search needs a sorted list
# binary search is very fast: 9 to 10 searches needed only



# babble sort
# straight exchange sort

# define function for babble sort
def babblesort(l):
    n = len(l)

    for i in range(1, n, 1):
        for j in range(n - 1, i - 1, -1):
            if l[j] < l[j - 1]:
                # swap
                l[j], l[j - 1] = l[j - 1], l[j]
                # we do a straight exchange sort

    return l

# main program
l = [1, 0, -8, 4, -3, 6, 7, 5, -5]
print(l)

# l.sort()
# print(l)

# we use: help(list)
# use "help(list)" to read description

l = babblesort(l)
print(l)



# insertion sort

# define the function for insertion sort
def insertionsort(array):
    for i in range(1, len(array)):
        v = array[i]
        j = i

        # compare array[j - 1] and array[i]
        while j > 0 and array[j - 1] > v:
            array[j] = array[j - 1]
            j = j - 1

        array[j] = v

    return array

# v is equal to array[i]
# insertion sort compares array[i] and array[j]

# insertion sort uses a sorted list and an unsorted list
# insertion sort keeps the sorted list always sorted

# main program

#l = [1, 0, -8, 4, -3, 6, 7, 5, -5]
l = [60, 38, 98, 54, 32, 90, 20]
print(l)

# l.sort()
# print(l)

l = insertionsort(l)
print(l)



# define babble sort
def babblesort(l):
    n = len(l)

    for i in range(1, n, 1):
        for j in range(n - 1, i - 1, -1):
            if l[j] < l[j - 1]:
                # swap
                l[j], l[j - 1] = l[j - 1], l[j]

    return l

# we now input two lists
def babblesort2(l, l2):
    n = len(l)

    for i in range(1, n, 1):
        for j in range(n - 1, i - 1, -1):
            if l[j] < l[j - 1]:
                # swap
                l[j], l[j - 1] = l[j - 1], l[j]
                # swap
                l2[j], l2[j - 1] = l2[j - 1], l2[j]

    return l, l2

# main program
l1 = []
l2 = []

l1.append(str(input('Name: ')))
l2.append(int(input('Score: ')))

l1.append(str(input('Name: ')))
l2.append(int(input('Score: ')))

l1.append(str(input('Name: ')))
l2.append(int(input('Score: ')))

l1.append(str(input('Name: ')))
l2.append(int(input('Score: ')))

l1, l2 = babblesort2(l2, l1)
print(l2)



# we use: https://docs.python.org/3.6/library/index.html
# use: https://www.springboard.com/blog/data-science-interview-questions/#programming

# https://www.springboard.com/blog/python-interview-questions/
# we use: https://docs.python.org/2/library/stdtypes.html

import numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Gaussian distribution mean and variance
mu, sigma = 100, 15

x = mu + sigma*np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)

l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')

plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])

plt.grid(True)
plt.show()



# use: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we use: https://news.efinancialcareers.com/dk-en/285249/machine-learning-and-big-data-j-p-morgan
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# use: https://www.jpmorgan.com/global/research/machine-learning

# we now use: https://www.jpmorgan.com/global/research/machine-learning
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we use LASSO regression
# L1 regularisation, L1 penalty term

# weight decay using LASSO
# we use the L2 norm as a penalty term

# Lagrange multiplier from the validation set
# use: (function to minimize) + (Lagrange multiplier) * (L1 term)

# we import Lasso regression
from sklearn.linear_model import Lasso

# we use LASSO regression
model = Lasso(alpha=0.1)

# we fit the model
model.fit([[-1,-1], [0,0], [1,1]], [-1,0,1])

print("")
print(model.coef_)

print(model.intercept_)

print(model.predict([[3, -3]]))
print("")

# we now use Ridge regression
# Ridge => L2 regularisation

# we now use the L2 norm as a penalty term
# use: (function to minimize) + (Lagrange multiplier) * (L2 term)

# L1 regularisation is better than L2 regularisation
# SOS: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we import Ridge regression
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)
model.fit([[-1,-1], [0,0], [1,1]], [-1,0,1])

print(model.coef_)

print(model.intercept_)

print(model.predict([[3, -3]]))
print("")

# we now use elastic net regression

# elastic net has both L1 and L2 penalty terms
# elastic net uses both L1 and L2 regularisation

# elastic net is between L1 and L2
# elastic net is between LASSO and Ridge

# use: https://www.jpmorgan.com/global/research/machine-learning
# elastic net: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# import elastic net
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1)
model.fit([[-1,-1], [0,0], [1,1]], [-1,0,1])

print(model.coef_)

print(model.intercept_)

print(model.predict([[3, -3]]))
print("")

# use K-Nearest Neighbors

# we import K-Nearest Neighbors
from sklearn.neighbors import NearestNeighbors

import numpy as np

# define numpy np array
X = np.array([[-1, -2], [-2,-2], [-3,-5], [1,1], [2,2], [4,4]])

model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = model.kneighbors([[0,0]])

print(distances)
print(indices)
print("")

# we use: https://news.efinancialcareers.com/dk-en/285249/machine-learning-and-big-data-j-p-morgan
# use: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# binary classification
# logistic regression for classification

# we import logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2')

model.fit([[-2,-3], [1,0], [1,1]], [1,0,1])

print(model.coef_)

print(model.intercept_)

print(model.predict([[3, 3]]))
print("")

# classification using SVMs
# we use SVMs, convex optimization problems

# kernel trick => higher dimensional space
# with kernel trick: linearly separable

# SVM for multi-class classification

# import SVM classifier
from sklearn.svm import SVC

import numpy as np

X = np.array([[-3,-2], [-4,-5], [3,4], [4,5]])
y = np.array([1, 1, 2, 2])

model = SVC()
model. fit(X, y)

print(model.predict([[0, 0]]))
print("")

# use: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# use clustering with K-means
# clustering is unsupervised learning

# we import K-means
from sklearn.cluster import KMeans

# we fit the model
model = KMeans(n_clusters=2, random_state=0).fit(X)

print(model.predict([[1,2], [-1,-1]]))

print(model.cluster_centers_)
print("")

# we now use PCA: principal component analysis
# use eigenvalues and eigenvectors for PCA

# PCA for dimensionality reduction
# dimensionality reductsion is unsupervised learning

# PCA is unsupervised learning
# we reduce the data: reduce the dimensions of the data

# we import PCA
from sklearn.decomposition import PCA

X = np.array([[-3,-2], [-4,-5], [3,4], [4,5]])

# we define the number of principal components
model = PCA(n_components=2)

# we do unsupervised learning
model.fit(X)

# check the variance and see if it is a good fit
print(model.explained_variance_ratio_)
print("")

# factor analysis and PCA
# PCA and ICA for dimensionality reduction

# we use: https://www.jpmorgan.com/global/research/machine-learning

# use: https://news.efinancialcareers.com/dk-en/285249/machine-learning-and-big-data-j-p-morgan
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

