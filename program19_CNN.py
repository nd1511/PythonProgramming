# we use Keras
from keras import layers
# use keras layers

# use keras models
from keras import models

# Data Science, Programming
# use: https://www.springboard.com/blog/data-science-interview-questions/#programming

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',

input_shape = (150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

#model.summary()
#print(model.summary())

# use keras optimizers
from keras import optimizers

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# model.summary()
print(model.summary())



# we now use dropout to reduce overfitting

# dropout, we use dropout
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

# use dropout
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# model.summary()
print(model.summary())



import os, shutil

original_dataset_dir = '/Users/dionelisnikolaos/Downloads/kaggle_original_data'

base_dir = '/Users/dionelisnikolaos/Downloads/cats_and_dogs_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)



fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]

for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_dogs_dir, fname)

    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))

print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))
print('')



# Python Online course: Interactive Python problems.
# http://interactivepython.org/runestone/static/pythonds/index.html

# Use the book as a roadmap: http://interactivepython.org/runestone/static/pythonds/index.html
# Recursion with finite memory and stack. Trees, graphs.

# use: https://www.springboard.com/blog/data-science-interview-questions/#programming

# List of interview questions:
# www.github.com/MaximAbramchuck/awesome-interview-questions

# The main website for Python coding questions:
# https://www.springboard.com/blog/data-science-interview-questions/#programming

#### Define a function in Python:
####    def partitions(amount, coins)
####
#### where amount is an integer and coins is a list of positive integers. The list coins
#### are the coins we have available and amount is the amount of money that we need.
####
#### The function should return the different ways the amount can be written as an addition of the coins.
#### Addition x+y is equal to y+x.
####
#### If amount <= 0 or if coins is empty, then the function should return 0.
####
#### *** Example ***
#### If coins =  [1, 2, 5] and amount = 5, then 4 ways:
#### 5  1+1+1+1+1  1+2+2  1+1+1+2.
####

# Coins = eval(input("Please input the elements of coins: "))
# Amount = int(input("Please input amount: "))

# A Python program to print all permutations of given length
from itertools import permutations

# Get all permutations of length 2 and length 2
perm = permutations([1, 2, 3], 2)

# Print the obtained permutations
for i in list(perm):
    print(i)

def partitions(amount, coins):
    if amount <= 0 or len(coins) == 0:
        return 0

    sum1 = 0
    sum3 = []

    for i in coins:
        amount2 = amount
        sum2 = []

        while amount2 > 0:
            amount2 -= i
            sum2.append(i)

        if amount2 == 0:
            #sum1 += 1

            # sum3.append(sum2)
            sum2.sort()
            if sum2 not in sum3:
                sum3.append(sum2)
                sum1 += 1
        else:
            amount2 += i
            sum2.pop()

            for j in coins:
                if i != j:
                    while amount2 > 0:
                        amount2 -= j
                        sum2.append(j)

                    if amount2 == 0:
                        #sum1 += 1

                        #sum3.append(sum2)
                        sum2.sort()
                        if sum2 not in sum3:
                            sum3.append(sum2)
                            sum1 += 1
                    else:
                        amount2 += j
                        sum2.pop()

    for k in sum3:
        for kk in range(1,len(k)):
            if k[kk-1]+k[kk] in coins:
                list1 = [k[kk-1]+k[kk]]
                list1.extend(k[:kk-1])
                list1.extend(k[kk+1:])

                #sum3.append(list1)
                list1.sort()
                if list1 not in sum3:
                    sum3.append(list1)
                    sum1 += 1

    #for i in coins:
    #    amount2 = amount
    #    while amount2 > 0:
    #        amount2 -= i

    return sum1, sum3

print('')
print(partitions(5, [1, 2, 5]))
print(partitions(5, [1, 2, 3, 5]))

print('')
Coins = eval(input("Please input the elements (at least two values) of coins: "))
Amount = int(input("Please input amount: "))

print('')
print(partitions(Amount, Coins))



# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# website: https://www.springboard.com/blog/data-science-interview-questions/#programming
# we now use: https://www.springboard.com/blog/data-science-interview-questions/

#### Exercise:
####
#### Return the numbers that occur an odd number of times in L.
#### If L = [ [1, 2, 3], [3, 1], [6], [8, 7, 6] ], then result = [ 2, 7, 8 ].
####
#### L = eval(input("L = : "))

def numbersOccurringOddNumberOfTimes(list1):
    mainList = []

    for i in list1:
        for j in i:
            if j not in mainList:
                mainList.append(j)
            else:
                mainList.remove(j)

    #return mainList
    mainList.sort()
    return mainList

L = [ [1, 2, 3], [3, 1], [6], [8, 7, 6] ]
print('')

#numbersOccurringOddNumberOfTimes(.)
print(numbersOccurringOddNumberOfTimes(L))

L = eval(input("L = : "))
print(numbersOccurringOddNumberOfTimes(L))



# use: https://www.springboard.com/blog/data-science-interview-questions/
# https://www.springboard.com/blog/data-science-interview-questions/#programming

# we use: http://interactivepython.org/runestone/static/pythonds/index.html#

#### dictionary dict likes
#### dict has keys and values
#### keys = names of boys and values are list of names of girls (unique names)
####
#### unique names hence set
#### set(.), list(set(.))
####
#### Python function
####    liked(likes)
####
#### Example:
####  likes = {"Michael": ["Maria", "Helen"],
####       "John": ["Maria"],
####       "Manos": ["Helen", "Katerina", "Maria"],
####       "Costas": ["Joana"],
####      }
####
#### then:
#### {
####   "Maria": ["John", "Manos", "Michael"],
####   "Helen": ["Manos", "Michael"],
####   "Katerina": ["Manos"],
####   "Joana": ["Costas"],
#### }
####

def liked(likes):
    likes2 = {}

    for i in likes:
        for j in likes[i]:
            if j not in likes2:
                likes2[j] = [i]
            else:
                likes2[j].append(i)

    return likes2

likes = {"Michael": ["Maria", "Helen"], \
    "John": ["Maria"], \
    "Manos": ["Helen", "Katerina", "Maria"], \
    "Costas": ["Joana"]}

print('')
print(liked(likes))

# Data Science, Programming
# use: https://www.springboard.com/blog/data-science-interview-questions/#programming

# recursion, memoization, maze problems
# http://interactivepython.org/runestone/static/pythonds/Recursion/ExploringaMaze.html

# we now use: http://interactivepython.org/runestone/static/pythonds/index.html
# use: http://interactivepython.org/runestone/static/pythonds/Recursion/ExploringaMaze.html

# Data Science (DS), Machine Learning
# DS: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

#### Exercise
####
#### People with names 0, 1, ..., N-1 for 1 day. List of length N.
#### People in the list: first, second, ..., last. List = who.
#### who[0] = first, who[1] = last
####
#### The list time (of length N) has the time spent for each person/customer.
#### time[0] = time spent for the first person, time[1] = time spent for the second person
####
#### The list start (of length N) find the start time for people[i]
#### The list over (of length N) find the end time for people[i]
#### 
#### *** Example ***
#### If who = [2, 4, 3, 0, 1] and time = [10, 10, 15.5, 5.3, 8],
#### then start = [13.3, 38.8, 23.3, 0, 5.3] and over = [23.3, 48.8, 38.8, 5.3, 13.3].
####
#### N=5
#### who = [3, 4, 2, 0, 1]
#### time = [10, 10, 15.5, 5.3, 8]
#### print("Η λίστα who: {}".format(who))
#### print("Η λίστα time: {}".format(time))

import numpy as np

def startAndOver(who, time):
    start = np.zeros(len(who))
    over = np.zeros(len(who))

    for i in range(len(who)):
        if i == 0:
            start[who.index(0)] = 0
            over[who.index(0)] = time[who.index(0)]
        else:
            start[who.index(i)] = start[who.index(i-1)] + time[who.index(i-1)]
            over[who.index(i)] = over[who.index(i-1)] + time[who.index(i)]

    return start, over

N = 5
who = [2, 4, 3, 0, 1]
time = [10, 10, 15.5, 5.3, 8]

print('')
print("List who: {}".format(who))
print("List time: {}".format(time))

print('')
print(startAndOver(who, time))

who = [2, 4, 5, 6, 3, 0, 1]
time = [10, 10, 15.5, 14.2, 1.3, 5.3, 8]

print('')
print(startAndOver(who, time))



####  Define a function
####    fof(D, name)
####
#### where D is a dictionary whose keys are names (strings). The value of each key k is a
#### list with the name of all the friends of k. The parameter name is a name (string).
####
#### The function returns a list with the people appearing in D and are
#### the friends of the friends of name but not the friends of name.
####
####  * name should not be in the returned list
####  * All names should be returned one time. => set(.), list(set(.))
####  * The returned list should be sorted inversely.
####
#### Friendship is not symmetrical: A can be within the friends of B but B
#### not be within the friends of A (like in the example below).
#### 
#### *** Example ***
####  D = {"Mihalis": ["Yannis", "Manolis"],
####       "Yannis": ["Mihalis", "Kostas", "Manolis"],
####       "Manolis": ["Kostas", "Dimitris"],
####       "Kostas": ["Yannis", "Giorgos"]
####      }
#### 
#### and name="Mihalis", then the returned list is ["Kostas", "Dimitris"].

def fof(D, name):
    list1 = []

    for i in D[name]:
        for j in D[i]:
            if j != name and j not in D[name]:
                list1.append(j)

    list1 = list(set(list1))

    #list1.sort()
    #list1.reverse()
    list1 = sorted(list1, reverse=True)

    return list1

D = {"Mihalis": ["Yannis", "Manolis"], \
    "Yannis": ["Mihalis", "Kostas", "Manolis"], \
    "Manolis": ["Kostas", "Dimitris"], \
     "Kostas": ["Yannis", "Giorgos"]}

print('')
print(fof(D, name="Mihalis"))

print('')
print("The dict is: {}".format(D))

print('')
name = input("Please input a name: ")
print(fof(D, name))

# use: https://www.springboard.com/blog/data-science-interview-questions/#programming
# https://leonmercanti.com/books/personal-development/Cracking%20the%20Coding%20Interview%20189%20Programming%20Questions%20and%20Solutions.pdf

# we also use: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# Python Online course: Interactive Python problems.
# we use: http://interactivepython.org/runestone/static/pythonds/index.html

# Recursion with finite memory and stack. Memoization. Trees, graphs.
# List of interview questions: www.github.com/MaximAbramchuck/awesome-interview-questions

# Website for Python coding questions: https://www.springboard.com/blog/data-science-interview-questions/#programming



#### dict friends with keys and values
#### where keys are people (strings) and values are his friends (strings)
####
#### Given a name, find the list of people in the dict that have name as friend.
#### sort(.), sorted
####
#### If friends is the dict shown below and name = "Manolis",
#### then result=["Giorgos"], and if name = "Giorgos" then result=[].
#### If name = "Maria" then result=["Eleni", "Giorgos", "Mihalis"].

# define findFriends
def findFriends(friends, name):
    list1 = []

    for i in friends:
        if name in friends[i]:
            list1.append(i)

    list1.sort()
    return list1

# main program

friends = {"Manolis": ["Mihalis", "Yannis"], \
    "Maria": ["Mihalis", "Eleni"], \
    "Eleni": ["Mihalis", "Maria"], \
    "Mihalis": ["Maria"], \
    "Yannis": [], \
    "Giorgos": ["Manolis", "Eleni", "Maria"]}

print('')
print("The dict is: {}".format(friends))

print('')
print(findFriends(friends, "Manolis"))

print(findFriends(friends, "Giorgos"))
print(findFriends(friends, "Maria"))

print('')
name = input("Please input a name: ")
print(findFriends(friends, name))

# we now use: https://www.springboard.com/blog/data-science-interview-questions/#programming
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf



#### Define a function
####   symdiff(A, B)
####
#### where lists A και B.
#### The function returns a list that contains elements only in one of the lists A and B.
####
#### The returned list does not contain double elements.
#### set(.), list(set(.))
####
#### The returned list does not need to be ordered.

def symdiff(A, B):
    list1 = []

    for i in A:
        if i not in B:
            list1.append(i)

    for i in B:
        if i not in A:
            list1.append(i)

    list1 = list(set(list1))
    return list1

def symdiff2(A, B):
    list1 = []

    for i in A:
        if i not in B and i not in list1:
            list1.append(i)

    for i in B:
        if i not in A and i not in list1:
            list1.append(i)

    #list1 = list(set(list1))
    return list1

# main program
A = [1,2,3,4,5]
B = [3,4,5,6,7,8,9]

print('')
print(symdiff(A, B))

A = [1,2,1,2,2]
B = [2,3,2,3,2,3,3,4]

print('')
print(symdiff(A, B))
print(symdiff2(A, B))

print('')
A = eval(input("Please insert list A: "))
B = eval(input("Please insert list B: "))

print('')
print(symdiff(A, B))

# we now use: https://www.springboard.com/blog/data-science-interview-questions/#programming
# https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we use: https://www.springboard.com/blog/python-interview-questions/

# What is the difference between a tuple and a list in Python?
# Apart from tuples being immutable there is also a semantic distinction that should guide
# their usage. => https://www.springboard.com/blog/data-science-interview-questions/#programming

# Algothon 2018, Reuters NLP Challenge: http://algothon.org
# https://www.kaggle.com/c/algothon-2018

# use: https://www.kaggle.com/c/algothon-2018/data
# we use: https://www.kaggle.com/c/algothon-2018

# we now use: https://algothon-2018.devpost.com/?ref_content=default&ref_feature=challenge&ref_medium=discover
# algorithmic trading: machine learning and algorithmic trading, http://algothon.org

# deep learning and machine learning
# we use: https://www.deeplearningbook.org

# Compute the sum 1/2 + 3/5 + 5/8 + .... for N terms with recursion and with no recursion.
# Hence: 1/2 + 3/5 + 5/8 + 7/11 + 9/14 + ....
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
print(function2(9))
print('')

# Algothon 2018, Reuters NLP Challenge: http://algothon.org
# https://www.kaggle.com/c/algothon-2018

# use: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/BasicDS/toctree.html

# we use one-line lambda expressions in Python
# use: https://docs.python.org/2/reference/expressions.html#lambda

# website: https://www.w3resource.com/c-programming-exercises/recursion/index.php

# we use: https://docs.python.org/2/reference/expressions.html
# website: https://docs.python.org/2/reference/expressions.html#lambda

