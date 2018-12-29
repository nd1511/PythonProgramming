# use: https://drive.google.com/open?id=1sbnRuR26hEXK8WXlJCIQFnndPZI2M1nx
# we use: https://drive.google.com/open?id=1NLtR4uMVtxojIWXZOdU2jPQZrtuzNC5c

# use: https://news.efinancialcareers.com/dk-en/285249/machine-learning-and-big-data-j-p-morgan
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we use: https://www.jpmorgan.com/global/research/machine-learning
# use: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# Quandl: import quandl, https://www.quandl.com/tools/python
# Bloomberg: BLPAPI, https://www.bloomberglabs.com/api/libraries/

# Webscraping: BeautifulSoup, import bs4, PyPi
# Webscraping: import Selenium, PyPi

# Twitter: import twitter, PyPi
# LinkedIn: import python-linkedin, PyPi

# we now use: https://drive.google.com/open?id=1sbnRuR26hEXK8WXlJCIQFnndPZI2M1nx
# we use: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# Quandl: import quandl, https://www.quandl.com/tools/python
import quandl

# LinkedIn: import python-linkedin, PyPi
import linkedin

# Webscraping: import BeautifulSoup, PyPi
import bs4

# do webscraping: import Selenium, PyPi
import selenium

# we use: https://drive.google.com/open?id=1sbnRuR26hEXK8WXlJCIQFnndPZI2M1nx

# we use pandas
import pandas as pd
#import pandas_datareader.data as web

import datetime
from numpy import nan

# we use selenium
from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# we create a pandas dataframe to store the scraped data
df = pd.DataFrame(index=range(40), columns=['company', 'quarter', 'quarter_ending', 'total_revenue', \
                                            'gross_profit', 'net_income', 'total_assets', 'total_liabilities', \
                                            'total_equity', 'net_cash_flow'])

url_form = "http://www.nasdaq.com/symbol/{}/financials?query={}&data=quarterly"
financials_xpath = "//tbody/tr/th[text() = '{}']/../td[contains(text(), '$')]"

# define the company ticker symbols
symbols = ["amzn", "aapl", "fb", "ibm", "msft"]

# file: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# Webscraping: BeautifulSoup, import bs4, PyPi
from bs4 import BeautifulSoup

import re
import os

import pandas as pd
from pandas import ExcelWriter

import sys

# we use: urllib.request
import urllib.request
# The urllib2 module has been split across two modules in Python 3 named urllib.request and urllib.error.

# from urllib.request, we use ProxyHandler
proxy_support = urllib.request.ProxyHandler({"https" : "https://proxy.companyname.net:8080"})

opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

# use: https://news.efinancialcareers.com/dk-en/285249/machine-learning-and-big-data-j-p-morgan
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we use: https://www.jpmorgan.com/global/research/machine-learning
# use: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# Vazirani Algorithms, Book
# use: http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf

# https://github.com/haseebr/competitive-programming/blob/master/Materials/Algorithms%20by%20Vazirani%20and%20Dasgupta.pdf
# we now use: http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf

# Cracking the Coding Interview, Book
# use: https://github.com/jwang5675/ctci/blob/master/Cracking%20the%20Coding%20Interview%206th%20Edition.pdf

# we use: http://faculty.washington.edu/pisan/cpp/readings/McDowell_ch11_Advanced%20Topics.pdf

# we use HackerRank
# use: https://www.hackerrank.com/nd1511
# nd1511, Nikolaos Dionelis, HackerRank

# Data Science GitHub: https://www.analyticsvidhya.com/blog/2018/12/best-data-science-machine-learning-projects-github
# generative adversarial networks (GANs): https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/

# use: https://www.analyticsvidhya.com/blog/2018/12/best-data-science-machine-learning-projects-github

# use: https://sigmoidal.io/machine-learning-for-trading/
# we use: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2756331

# we now use: https://drive.google.com/open?id=1NLtR4uMVtxojIWXZOdU2jPQZrtuzNC5c

# use PyTorch
import torch

x = torch.rand(5,3)
print(x)

x = torch.zeros(5, 3)
print(x)

x = torch.FloatTensor([5.5, 3])
print(x)

y = torch.rand(2)
print(torch.add(x, y))

# we use: https://docs.python.org/3.6/library/index.html
# use: https://docs.python.org/3.6/library/stdtypes.html#sequence-types-list-tuple-range

# we use babble sort
# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# we define babble sort
def babblesort(l):
    n = len(l)
    for i in range(1, n, 1):
        for j in range(n - 1, i - 1, -1):
            if l[j] < l[j - 1]:
                # swap
                l[j], l[j - 1] = l[j - 1], l[j]

    return l

# main program
names = []

while True:
    name = input('Write your name: ')

    if name == '0':
        break

    names.append(name)

print('')
print(names)

# help(list)
# names.sort()

# names = babblesort(names)
babblesort(names)

print(names)



# we use: http://interactivepython.org/runestone/static/pythonds/index.html#

# www.interactivepython.org
# Problem Solving with Algorithms and Data Structures using Python - interactivepython.org

# interactive textbook
# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# recursion and memoization
# Fibonacci series and memoization
# http://interactivepython.org/runestone/static/pythonds/index.html#

# we use babble sort
def modifiedbabblesort(l, l2):
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

scores = [41, 51, 31, 61, 81, 72]
names = ['a', 'c', 'b', 'f', 'd', 'b']

print('')
print(scores)
print(names)

# scores, names = modifiedbabblesort(scores, names)
modifiedbabblesort(scores, names)

print(scores)
print(names)

# help(list)
scores.reverse()
names.reverse()

print(scores)
print(names)



# Vazirani's Book: Algorithms
# use: http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf

# https://github.com/haseebr/competitive-programming/blob/master/Materials/Algorithms%20by%20Vazirani%20and%20Dasgupta.pdf
# we now use: http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf

# define insertion sort
def insertionsort(array):
    for i in range(1, len(array)):
        v = array[i]
        j = i

        while j > 0 and array[j - 1] > v:
            array[j] = array[j - 1]
            j = j - 1
        array[j] = v

    return array

# main program
names = []

while True:
    name = input('Write your name: ')

    if name == '0':
        break

    names.append(name)

print('')
print(names)

# help(list)
# names.sort()

# names = insertionsort(names)
insertionsort(names)

print(names)

# use: https://docs.python.org/3/library/functions.html

# we use the Python built-in functions
# https://docs.python.org/3/library/functions.html

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



# we use the Python documentation
# we use: https://docs.python.org/3/library/index.html

# we use: help(list)
# use: https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range

# main program
list1 = []

while True:
    num = int(input('Write a number: '))
    list1.append(num)

    if sum(list1) > 1000:
        break

print('')
print(list1)

print(sum(list1))
print(max(list1))

# max_value = max(list1)
# max_index = my_list.index(max_value)

# help(list)
print(list1.count(max(list1)))

# help(str) and help(list)
# use: http://interactivepython.org/runestone/static/pythonds/Introduction/GettingStartedwithData.html

# Fibonacci series
# recursion and memoization

# define the recursion for the Fibonacci series
def Fib(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    elif n > 1:
        return Fib(n-1) + Fib(n-2)

# main program
print('')

print(Fib(4))
print(Fib(5))

# we use: https://www.youtube.com/watch?v=Qk0zUZW-U_M
# http://interactivepython.org/runestone/static/pythonds/index.html#

# Fibonacci series and memoization
# http://interactivepython.org/runestone/static/pythonds/index.html#

# we store the values
Fib_cache = {}

def Fib2(n):
    if n in Fib_cache:
        return Fib_cache[n]

    if n == 0:
        return 1
    elif n == 1:
        return 1
    elif n > 1:
        value1 = Fib2(n-1) + Fib2(n-2)

        # store the value
        Fib_cache[n] = value1

        # return the value
        return value1

# main program
print('')

for n in range(0, 10+1, 1):
    print(n, ' term: ', Fib2(n))

# use memoization
from functools import lru_cache

# we store the first 1000 values
lru_cache(maxsize = 1000)

print('')
for n in range(0, 10+1, 1):
    print(n, ' term: ', Fib(n))



# we use: https://docs.python.org/3.6/library/index.html
# use: https://docs.python.org/3.6/library/stdtypes.html#sequence-types-list-tuple-range

# define function to separate list
def separateList(list1):
    list2 = []
    list3 = []

    for i in list1:
        if i > 0:
            list2.append(i)
        elif i < 0:
            list3.append(i)

    return list2, list3

# main program
listTest = [4, 3, 0, 0, -3, -6]

print('')
print(separateList(listTest))

import numpy as np

print(listTest > 0 * np.ones(len(listTest)))
print(listTest < 0 * np.ones(len(listTest)))



# we use HackerRank
# nd1511, Nikolaos Dionelis, HackerRank
# use: https://www.hackerrank.com/nd1511

# main program

import cmath
import math

x1 = -1.0
y1 = 1.0

x2 = -2.0
y2 = 10.0

point1 = complex(x1, y1)
point2 = complex(x2, y2)

# use: phase(point1)
# use: phase(point2)

print('')
print(math.degrees(cmath.phase(point1)))
print(math.degrees(cmath.phase(point2)))

result1 = False

phase1 = math.degrees(cmath.phase(point1))
phase2 = math.degrees(cmath.phase(point2))

if (0 < phase1 < 90 and 0 < phase2 < 90) or (90 < phase1 < 180 and 90 < phase2 < 180) or \
        (-180 < phase1 < -90 and -180 < phase2 < -90) or (-90 < phase1 < 0 and -90 < phase2 < 0):
    result1 = True

print(result1)



# main program

A = [1, 2]
B = [-1, -1, -1, 10]
C = [11]

print('')

print(sum(A + B + C))
print(sum(set(A + B + C)))

# result1 = sum(A+B+C)
result1 = sum(set(A + B + C))

print(result1)
print('')

A = [1, 2]
B = ["a", "b"]
C = [1.1, "x"]

result = []

for i in range(0, min([len(A), len(B), len(C)]), 1):
    result.append(A[i])
    result.append(B[i])
    result.append(C[i])

value1 = i

indexMin = [len(A), len(B), len(C)].index(min([len(A), len(B), len(C)]))

if indexMin == 0:
    for i in range(value1 + 1, min([len(B), len(C)]), 1):
        result.append(B[i])
        result.append(C[i])

    value2 = i

    indexMin2 = [len(B), len(C)].index(min([len(B), len(C)]))

    if indexMin2 == 0:
        for i in range(value2 + 1, len(C), 1):
            result.append(C[i])

    elif indexMin2 == 1:
        for i in range(value2 + 1, len(B), 1):
            result.append(B[i])

elif indexMin == 1:
    for i in range(value1 + 1, min([len(A), len(C)]), 1):
        result.append(A[i])
        result.append(C[i])

    value2 = i

    indexMin2 = [len(A), len(C)].index(min([len(A), len(C)]))

    if indexMin2 == 0:
        for i in range(value2 + 1, len(C), 1):
            result.append(C[i])

    elif indexMin2 == 1:
        for i in range(value2 + 1, len(A), 1):
            result.append(A[i])

elif indexMin == 2:
    for i in range(value1 + 1, min([len(A), len(B)]), 1):
        result.append(A[i])
        result.append(B[i])

    value2 = i

    indexMin2 = [len(A), len(B)].index(min([len(A), len(B)]))

    if indexMin2 == 0:
        for i in range(value2 + 1, len(B), 1):
            result.append(B[i])

    elif indexMin2 == 1:
        for i in range(value2 + 1, len(A), 1):
            result.append(A[i])

print(result)



# we define the function functionList
def functionList(A, B, C):
    result = []

    for i in range(0, min([len(A), len(B), len(C)]), 1):
        result.append(A[i])
        result.append(B[i])
        result.append(C[i])

    value1 = i

    indexMin = [len(A), len(B), len(C)].index(min([len(A), len(B), len(C)]))

    if indexMin == 0:
        for i in range(value1 + 1, min([len(B), len(C)]), 1):
            result.append(B[i])
            result.append(C[i])

        value2 = i

        indexMin2 = [len(B), len(C)].index(min([len(B), len(C)]))

        if indexMin2 == 0:
            for i in range(value2 + 1, len(C), 1):
                result.append(C[i])

        elif indexMin2 == 1:
            for i in range(value2 + 1, len(B), 1):
                result.append(B[i])

    elif indexMin == 1:
        for i in range(value1 + 1, min([len(A), len(C)]), 1):
            result.append(A[i])
            result.append(C[i])

        value2 = i

        indexMin2 = [len(A), len(C)].index(min([len(A), len(C)]))

        if indexMin2 == 0:
            for i in range(value2 + 1, len(C), 1):
                result.append(C[i])

        elif indexMin2 == 1:
            for i in range(value2 + 1, len(A), 1):
                result.append(A[i])

    elif indexMin == 2:
        for i in range(value1 + 1, min([len(A), len(B)]), 1):
            result.append(A[i])
            result.append(B[i])

        value2 = i

        indexMin2 = [len(A), len(B)].index(min([len(A), len(B)]))

        if indexMin2 == 0:
            for i in range(value2 + 1, len(B), 1):
                result.append(B[i])

        elif indexMin2 == 1:
            for i in range(value2 + 1, len(A), 1):
                result.append(A[i])

    return result

# main program

A = [1, 2]
B = ["a", "b"]
C = [1.1, "x"]

print(functionList(A, B, C))

A = [1, 2, 89, 31, 54, 67, 12]
B = ["a", "b", "d", "c"]
C = [1.1, "x"]

print(functionList(A, B, C))

A = [1, 2]
B = [-1, -1, -1, 10]
C = [11]

result = A + B + C
result.sort()

print('')
print(result)

import numpy as np

A = [1, 2]
B = [-1, -1, -1, 10]
C = [11]

# use: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.concatenate.html
result = np.concatenate((A, B, C), axis=0)

# we use: np.concatenate()
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.concatenate.html

result = list(result)
result.sort()

print(result)



# we define the select function
def select(A, B, C):
    list1 = []

    for i in A:
        for j in B:
            # for k in C:
            # if (i == j) and (i != k):
            if i == j:
                list1.append(i)

    for i in list1:
        for j in C:
            if i == j:
                list1.remove(i)

    list1.sort()
    list1 = set(list1)

    return list(list1)

# main program

A = [1, 2, 3, 7]
B = [2, 3, 3, 4, 7]
C = [4, 5, 6, 7]

print('')
print(select(A, B, C))

# we use randint
from random import randint

# A = randint(0, 1)
A = randint(0, 1000)

print('')
print(A)

found1 = False

B = randint(0, 1000)

top = 1000
bottom = 0

for i in range(0, 10, 1):
    if A == B:
        found1 = True
        break

    elif A > B:
        bottom = B

    else:
        top = B

    # B = randint(bottom, top)
    B = int(0.5 * (bottom + top))

    print(B)

print('')
print(found1)

# print(i)
print('The program found the number with ', str(i), ' tries.')



# Cracking the Coding Interview, Book
# use: https://github.com/jwang5675/ctci/blob/master/Cracking%20the%20Coding%20Interview%206th%20Edition.pdf
# we use: http://faculty.washington.edu/pisan/cpp/readings/McDowell_ch11_Advanced%20Topics.pdf

# The Merge Sort Algorithm
# Merge Sort is a recursive Divide and Conquer algorithm.

# we use: http://interactivepython.org/runestone/static/pythonds/index.html#

# we define Merge Sort
def mergeSort(alist):
    print("Splitting ", alist)

    if len(alist) > 1:
        mid = len(alist) // 2

        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i = 0
        j = 0
        k = 0

        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k] = lefthalf[i]
                i = i + 1
            else:
                alist[k] = righthalf[j]
                j = j + 1
            k = k + 1

        while i < len(lefthalf):
            alist[k] = lefthalf[i]
            i = i + 1
            k = k + 1

        while j < len(righthalf):
            alist[k] = righthalf[j]
            j = j + 1
            k = k + 1

    print("Merging ", alist)

# main program
alist = [54, 26, 93, 17, 77, 31, 44, 55, 20]

mergeSort(alist)
print(alist)

# Mergesort and Quicksort are both recursive Divide and Conquer algorithms.
# Merge sort is a recursive algorithm that continually splits a list in half.

# If the list is empty or has one item, it is sorted by definition (i.e. the base case).

# If the list has more than one item, we split the list and recursively
# invoke a merge sort on both halves.

# Once the two halves are sorted, the fundamental operation, called a merge,
# is performed. Merging is the process of taking two smaller sorted lists and
# combining them together into a single, sorted, new list.

# we use: http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf
# https://github.com/haseebr/competitive-programming/blob/master/Materials/Algorithms%20by%20Vazirani%20and%20Dasgupta.pdf
# we now use: http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf

# use: https://github.com/jwang5675/ctci/blob/master/Cracking%20the%20Coding%20Interview%206th%20Edition.pdf

# we use: http://interactivepython.org/runestone/static/pythonds/index.html
# use: http://interactivepython.org/runestone/static/pythonds/Introduction/GettingStartedwithData.html

names = []

while True:
    name = input('Write your name: ')

    if name == '0':
        break

    names.append(name)

print('')
print(names)

# help(list)
# names.sort()

# names = mergeSort(names)
mergeSort(names)

print(names)

