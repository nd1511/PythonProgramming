# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cracking the Coding Interview
# use: https://github.com/jwang5675/ctci/blob/master/Cracking%20the%20Coding%20Interview%206th%20Edition.pdf

# we use: http://faculty.washington.edu/pisan/cpp/readings/McDowell_ch11_Advanced%20Topics.pdf

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
    # in binary search, the ceiling moves

    # binary search => define the first and last
    # binary search => we will change the first and last

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



# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# global x,y
# use global memory variables

# global variables => change the memory in the main program
# global variables => use less memory

# define the function mysum()
def mysum(a):
    # define x,y as global variables
    global x, y

    # global vs local memory variables

    x = a * x
    y = y + a

    print('x in fuction = ', x)
    print('y in function = ', y)

    # return statement
    return x + y

# main program
x = int(input("give me the first number"))
y = int(input("give me the second number"))

print('x before function = ', x)
print('y before function = ', y)

a = 10
print(mysum(a))

print('x after function = ', x)
print('y after function = ', y)



import numpy as np

# use pandas for .csv files
import pandas as ps

#help(list)
# use "help(list)" to read description

# help(file)
# help(list)

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
#print([list1[-1:0:-1], list1[0]])
#print([list1[-1:0:-1], [list1[0]]])
print(list1[-1:0:-1] + [list1[0]])
print("")

print(list1[::-2])
print("")

#help(list)

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

# use dry run
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

# Recursive: The function calls itself

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

print("I am back in the main program.")
print("")

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



# !/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# main program

l = [1, 0, -8, 4, -3, 6, 7, 5, -5]
print(l)

# l.sort()
# print(l)

# we use: help(list)
# use "help(list)" to read description

babblesort(l)
print(l)



# insertion sort

# define function for insertion sort
def insertionsort(array):
    for i in range(1, len(array)):
        v = array[i]

        j = i

        while j > 0 and array[j - 1] > v:
            array[j] = array[j - 1]
            j = j - 1

        array[j] = v

# main program

#l = [1, 0, -8, 4, -3, 6, 7, 5, -5]
l = [60, 38, 98, 54, 32, 90, 20]
print(l)

# l.sort()
# print(l)

insertionsort(l)
print(l)



# !/usr/bin/env python3
# -*- coding: utf-8 -*-

def babblesort(l):
    n = len(l)

    for i in range(1, n, 1):
        for j in range(n - 1, i - 1, -1):
            if l[j] < l[j - 1]:
                # swap
                l[j], l[j - 1] = l[j - 1], l[j]

def babblesort2(l, l2):
    n = len(l)

    for i in range(1, n, 1):
        for j in range(n - 1, i - 1, -1):
            if l[j] < l[j - 1]:
                # swap
                l[j], l[j - 1] = l[j - 1], l[j]

                # swap
                l2[j], l2[j - 1] = l2[j - 1], l2[j]

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

babblesort2(l2, l1)
print(l1)
