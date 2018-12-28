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

# Webscraping: import Selenium, PyPi
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
        print('gia ton ', i, ' oro tis listas')
        v = array[i]
        j = i
        while j > 0 and array[j - 1] > v:
            print('pairno to ', j - 1, 'oro tis ipolipis listas kai sygrino tin timi toy me ayti toy', i,
                  'kai kano allagi timon')
            array[j] = array[j - 1]
            j = j - 1
        array[j] = v
        print('vazo stin thesi toy ', j, 'tin timi tou arxikou ', i, 'apo opou xekinisa')

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

