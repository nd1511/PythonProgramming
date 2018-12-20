
# use: https://github.com/phiconad/Algothon_Georgeteam

# we use: http://www.algothon.org
# https://github.com/phiconad/Algothon_Georgeteam

# http://www.commsp.ee.ic.ac.uk/~mandic/courses.htm
# http://www.econ.ohio-state.edu/dejong/note2.pdf

# we use: http://www.commsp.ee.ic.ac.uk/~mandic/courses.htm
# we also use: https://pdfs.semanticscholar.org/presentation/27ca/53acde0b5a7914cb042365ae4f05a2c1c0ce.pdf

# for MATLAB, we use: https://github.com/yalamarthibharat/TimeSeriesAnalysis

# we use pandas
import pandas as pd

# we use numpy
import numpy as np

# we use Quandl
import quandl
#quandl.ApiConfig.api_key = "bA8kfL-FkszhWuXM18Pe" #setting up your api-key

from PyPDF2 import PdfFileReader

import os
import math

#help(list)
# use "help(list)" to read description

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

# base case to end recursion

# functions: use def and return
def factorial(x):
    p = 1

    # use: range(StartValue, EndValue+1, StepSize)
    for i in range(1, x+1, 1):
        p = p * i

    print(x, "! = ", p)

factorial(n)

# recursive implementations take more memory and are slower
# recursive implementations are neater

# recursive implementations are more understandable

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



# we use Chris Chatfield's book
# we use: The Analysis of Time Series: An Introduction, Chris Chatfield, 6th edition (2004), Chapman & Hall / CRC.

# References:
# [1] C. Chatfield, "The Analysis of Time Series: An Introduction" 6th edition (2004), Chapman & Hall / CRC.

# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 2.7: Autocorrelation and the Correlogram, 6th edition (2004), Chapman & Hall / CRC.
# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 14.3: Examples, 6th edition (2004), Chapman & Hall / CRC.

# we use: https://www.crcpress.com/The-Analysis-of-Time-Series-An-Introduction-Sixth-Edition/Chatfield/p/book/9781584883173

# datasets can be found in (https://www.crcpress.com/The-Analysis-of-Time-Series-An-Introduction-Sixth-Edition/Chatfield/p/book/9781584883173)
# C. Chatfield: https://www.crcpress.com/The-Analysis-of-Time-Series-An-Introduction-Sixth-Edition/Chatfield/p/book/9781584883173



# we use C. Chatfield's book
# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 2.7: Autocorrelation and the Correlogram, 6th edition (2004), Chapman & Hall / CRC.

# we use equation (2.7)
# we use (2.7) from: Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 2.7: Autocorrelation and the Correlogram, 6th edition (2004), Chapman & Hall / CRC.

# we use c_k and r_k

# we use numpy
import numpy as np

#print(np.random.randn())
#print(np.random.randn(1,2))

array1 = np.random.randn(1,200)

#x = (?)
x = array1

#print(array1.size)
#N = x.size

N = x.size
#print(N)

#M = N / 10
#print(M)

import math
M = math.floor(N / 10)
#print(M)

#c_k = np.zeros(int(M))
c_k = np.zeros(M)

x_bar = 0

#print(x[0,0])
#print(x[0,1])

for t in range(1-1,N-1):
    x_bar = x_bar + (x[0,t])

x_bar = x_bar / N

for m in range(1-1, M):
    for t in range(1-1, N-m):
        c_k[m] = c_k[m] + ((x[0,t] - x_bar) * (x[0,t+m] - x_bar))

c_k = (1/N) * c_k

r_k = c_k / c_k[0]

#print(c_k)
#print(c_k.size)

#print(c_k.size)
#print(r_k.size)

# we find the auto-correlation coefficients
# we compute the auto-correlation function, the auto-correlation coefficients, the correlogram



# we use: http://www.algothon.org
# use: https://github.com/phiconad/Algothon_Georgeteam

# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 14.3: Examples, 6th edition (2004), Chapman & Hall / CRC.

# accordding to example 14.3, (1) c_k, (2) r_k, (3) one-step difference and repeat (1) and (2)
# we have done (1) and (2) and now we do (3)

#x = (?)
x = array1[0,1:] - array1[0,0:-1]
# x is the one-step difference

#print(np.size(array1))
#print(np.size(x))

#print(array1.size)
N = x.size

#print(N)
#print(array1.size)

#M = N / 10
# print(M)

#import math
M = math.floor(N / 10)

#c_k = np.zeros(int(M))
c_k = np.zeros(M)

x_bar = 0

#print(x[0,0])
#print(x[0,1])

for t in range(1-1,N-1):
    #x_bar = x_bar + (x[0,t])
    x_bar = x_bar + (x[t])

x_bar = x_bar / N

for m in range(1-1, M):
    for t in range(1-1, N-m):
        #c_k[m] = c_k[m] + ((x[0,t] - x_bar) * (x[0,t+m] - x_bar))
        c_k[m] = c_k[m] + ((x[t] - x_bar) * (x[t + m] - x_bar))

c_k = (1/N) * c_k

r_k = c_k / c_k[0]

#print(c_k)
#print(c_k.size)

#print(c_k.size)
#print(r_k.size)

# we find the auto-correlation coefficients
# we compute the auto-correlation function, the auto-correlation coefficients, the correlogram



# we now use: Yield (%) on British government securities

# Yield (%) on British government securities
yield_Br = [2.22, 2.23, 1.78, 1.72, 2.36, 2.41, 3.45, 3.29, 2.93, 2.92, 2.29, 2.66, 4.58, 4.76, 5.00, 4.74, 5.66, 5.42, 4.26, 4.02, 4.69, 4.72, 5.42, 5.31, 5.98, 5.91, 4.67, 4.81, 4.75, 4.91, 6.67, 6.52, 6.58, 6.42, 6.69, 6.51, 7.52, 7.41, 8.11, 8.18, 8.79, 8.63]

#x = (?)
#x = array1
x = yield_Br

#print(array1.size)
#N = x.size

#N = x.size
N = len(x)
#print(N)

# we use plt to plot figures
import matplotlib.pyplot as plt

plt.figure()

plt.plot(x, 'bo-', label='Yield')
#plt.plot((1:len(x))*(1/len(x))*21, x, 'bo-', label='Yield')

plt.title('Yield (%) on British government securities')
plt.legend()

plt.show()

#M = N / 10
#print(M)

import math
M = math.floor(N / 10)
#print(M)

#c_k = np.zeros(int(M))
c_k = np.zeros(M)

x_bar = 0

#print(x[0,0])
#print(x[0,1])

for t in range(1-1,N-1):
    x_bar = x_bar + (x[t])

x_bar = x_bar / N

for m in range(1-1, M):
    for t in range(1-1, N-m):
        c_k[m] = c_k[m] + ((x[t] - x_bar) * (x[t+m] - x_bar))

c_k = (1/N) * c_k

r_k = c_k / c_k[0]

#print(c_k)
#print(c_k.size)

#print(c_k.size)
#print(r_k.size)

# we find the auto-correlation coefficients
# we compute the auto-correlation function, the auto-correlation coefficients, the correlogram

plt.figure()

#plt.plot(r_k, 'bo-', label='auto-correlation coefficients')
plt.plot([1,2,3,4], r_k, 'bo-', label='auto-correlation coefficients')
#plt.plot((1:len(x))*(1/len(x))*21, x, 'bo-', label='Yield')

plt.plot([1,2,3,4], np.ones(np.size(r_k))*2*(1/np.sqrt(N)), 'ro--', label='2/sqrt(N)')
plt.plot([1,2,3,4], np.ones(np.size(r_k))*2*(-1)*(1/np.sqrt(N)), 'ro--', label='-2/sqrt(N)')

plt.title('The auto-correlation coefficients, the correlogram')
plt.legend()

plt.show()

# we see that the auto-correlation coefficients do not go to zero

# the auto-correlation coefficients do not come down quickly towards zero
# from the plot, we observe that the auto-correlation coefficients do not go to zero fast



# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 14.3: Examples, 6th edition (2004), Chapman & Hall / CRC.

# accordding to example 14.3, (1) c_k, (2) r_k, (3) one-step difference and repeat (1) and (2)
# we have done (1) and (2) and now we do (3)

#import numpy as np
yield_Br = np.asarray(yield_Br)

#x = (?)
#x = array1[0,1:] - array1[0,0:-1]
x = yield_Br[1:] - yield_Br[0:-1]
# x is the one-step difference

#print(np.size(yield_Br))
#print(np.size(x))

#print(array1.size)
N = x.size

#print(N)
#print(array1.size)

#M = N / 10
# print(M)

#import math
M = math.floor(N / 10)

#c_k = np.zeros(int(M))
c_k = np.zeros(M)

x_bar = 0

#print(x[0,0])
#print(x[0,1])

for t in range(1-1,N-1):
    #x_bar = x_bar + (x[0,t])
    x_bar = x_bar + (x[t])

x_bar = x_bar / N

for m in range(1-1, M):
    for t in range(1-1, N-m):
        #c_k[m] = c_k[m] + ((x[0,t] - x_bar) * (x[0,t+m] - x_bar))
        c_k[m] = c_k[m] + ((x[t] - x_bar) * (x[t + m] - x_bar))

c_k = (1/N) * c_k

r_k = c_k / c_k[0]

#print(c_k)
#print(c_k.size)

#print(c_k.size)
#print(r_k.size)

# we find the auto-correlation coefficients
# we compute the auto-correlation function, the auto-correlation coefficients, the correlogram

plt.figure()

#plt.plot(r_k, 'bo-', label='auto-correlation coefficients')
plt.plot([1,2,3,4], r_k, 'bo-', label='auto-correlation coefficients')
#plt.plot((1:len(x))*(1/len(x))*21, x, 'bo-', label='Yield')

plt.plot([1,2,3,4], np.ones(np.size(r_k))*2*(1/np.sqrt(N)), 'ro--', label='2/sqrt(N)')
plt.plot([1,2,3,4], np.ones(np.size(r_k))*2*(-1)*(1/np.sqrt(N)), 'ro--', label='-2/sqrt(N)')

plt.title('The auto-correlation coefficients, the correlogram')
plt.legend()

plt.show()

# we see that the auto-correlation coefficients go to zero

# the auto-correlation coefficients come down quickly towards zero
# from the plot, we observe that the auto-correlation coefficients go to zero fast






# we now use a cryptocurrency dataset

# we use another dataset
# we utilise a cryptocurrency dataset

# we use pandas
import pandas as pd

from pandas import read_csv
#from pandas import datetime

#def parser(x):
#    return datetime.strptime('190' + x, '%Y-%m')

#series = read_csv('/Users/dionelisnikolaos/Downloads/dataset_cryptocurrency.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# we create a dataframe
#df = pd.read_csv('./dataset_cryptocurrency.csv')
df = pd.read_csv('/Users/dionelisnikolaos/Downloads/dataset_cryptocurrency.csv')

# output first 5 rows in dataset
print(df.head())

# output first 9 rows in dataset
print(df.head(9))

# The cryptocurrency dataset contains data for 4 cryptocurrencies (Bitcoin, Ethereum, Ripple, Bitcoin-Cash)
# over a 5 year period 2013-2018. The provided data is all in $ USD and includes open/close, high/low, volume traded, spread and close ratio.
# The close ratio is defined as: Close Ratio = (Close - Low)/(High - Low)

# The default setting is to use the close ratio of the time frame as the input data.
# We compute the close ration using Close Ratio = (Close - Low)/(High - Low).
# Changing this to the open, the high or low can dramatically affect how the indicator moves and the analytical insight it provides.

# The open, high, low and close average (OHLC average) is the average of all these settings combined.
# There is no right or wrong setting for an indicator. Whether to the use the open, high, low or close or an average will depend on the insight the trader requires from the indicator.



# In the Excel file: Bitcoin (BTC) is from 2 to 1867
# BTC is from 2 to 1867
# ETH is from 1868 to 2902
# XRP is from 2903 to 4670
# BCH is from 4671 to 4989

# In the stored pandas array
# BTC is from 0 to 1865
# ETH is from 1866 to 2900
# XRP is from 2901 to 4668
# BCH is from 4669 to 4987

# output first 1865+1 rows in dataset
#print(df.head(1865+1))
# this has all BTC

# output first 1865+1+1 rows in dataset
#print(df.head(1865+1+1))
# this has all BTC except the last one

#BTC = df.head(1865+1)
#print(BTC)
#print(BTC.size)

#BTC = df[0:1865]
BTC = df[0:1865+1]
#print(BTC)

#ETH = df.head(2900+1)
#ETH = df.DataFrame({'symbol':'ETH'})
#print(ETH)

#ETH = df[1866:2900]
ETH = df[1866:2900+1]
#print(ETH)

#XRP = df.head(4668+1)
#XRP = df[2901:4668]
XRP = df[2901:4668+1]
#print(XRP)

#BCH = df.head(4987+1)
#BCH = df.tail(4987-4669)
#print(BCH)
#print(BCH.size)

#BCH = df[4669:4987]
BCH = df[4669:4987+1]
#print(BCH)

BCH_close_ratio = BCH.close_ratio
#print(BCH_close_ratio)

XRP_close_ratio = XRP.close_ratio
#print(XRP_close_ratio)

ETH_close_ratio = ETH.close_ratio
#print(ETH_close_ratio)

BTC_close_ratio = BTC.close_ratio
#print(BTC_close_ratio)



# we now plot the Close Ratios

# we plot BTC
plt.figure()

#plt.plot(BTC_close_ratio, 'bo-', label='BTC close_ratio')
#plt.plot(ETH_close_ratio, 'ro-', label='ETH close_ratio')
#plt.plot(XRP_close_ratio, 'ko-', label='XRP close_ratio')
#plt.plot(BCH_close_ratio, 'mo-', label='BCH close_ratio')

plt.plot(range(0, 100+1), BTC_close_ratio[0:100+1], 'bo-', label='BTC close_ratio')

#plt.plot(BTC_close_ratio[0:100+1], 'bo-', label='BTC close_ratio')
#plt.plot(ETH_close_ratio[0:100+1], 'ro-', label='ETH close_ratio')
#plt.plot(XRP_close_ratio[0:100+1], 'ko-', label='XRP close_ratio')
#plt.plot(BCH_close_ratio[0:100+1], 'mo-', label='BCH close_ratio')

plt.title('The close ratios.')
plt.legend()

plt.show()

# we plot ETH
plt.figure()

plt.plot(range(0, 100+1), ETH_close_ratio[0:100+1], 'ro-', label='ETH close_ratio')

plt.title('The close ratios.')
plt.legend()

plt.show()



# we use the last values

# we plot all together using the last values
plt.figure()

plt.plot(range(0, 100+1), BTC_close_ratio[-100-1:], 'bo-', label='BTC close_ratio')
plt.plot(range(0, 100+1), ETH_close_ratio[-100-1:], 'ro-', label='ETH close_ratio')
plt.plot(range(0, 100+1), XRP_close_ratio[-100-1:], 'ko-', label='XRP close_ratio')
plt.plot(range(0, 100+1), BCH_close_ratio[-100-1:], 'mo-', label='BCH close_ratio')

plt.title('The close ratios.')
plt.legend()

plt.show()

#print(len(BTC_close_ratio))
#print(len(ETH_close_ratio))
#print(len(XRP_close_ratio))
#print(len(BCH_close_ratio))

BTC_close_ratio2 = np.asarray(BTC_close_ratio[-len(BCH_close_ratio):])
ETH_close_ratio2 = np.asarray(ETH_close_ratio[-len(BCH_close_ratio):])
XRP_close_ratio2 = np.asarray(XRP_close_ratio[-len(BCH_close_ratio):])

#print(len(BTC_close_ratio2))
#print(len(ETH_close_ratio2))
#print(len(XRP_close_ratio2))
#print(len(BCH_close_ratio))

# we plot all together using the last values
plt.figure()

plt.plot(range(0, len(BCH_close_ratio)), BTC_close_ratio2, 'bo-', label='BTC close_ratio')
plt.plot(range(0, len(BCH_close_ratio)), ETH_close_ratio2, 'ro-', label='ETH close_ratio')
plt.plot(range(0, len(BCH_close_ratio)), XRP_close_ratio2, 'ko-', label='XRP close_ratio')
plt.plot(range(0, len(BCH_close_ratio)), BCH_close_ratio, 'mo-', label='BCH close_ratio')

plt.title('The close ratios.')
plt.legend()

plt.show()



# we compute the auto-correlation coefficients

# we compute the auto-correlation coefficients for BTC

#x = (?)
#x = array1

#x = BTC_close_ratio
x = BTC_close_ratio2

#print(array1.size)
#N = x.size

#N = x.size
N = len(x)
#print(N)

#M = N / 10
#print(M)

import math
M = math.floor(N / 10)
#print(M)

#c_k = np.zeros(int(M))
c_k = np.zeros(M)

x_bar = 0

#print(x[0,0])
#print(x[0,1])

for t in range(1-1,N-1):
    x_bar = x_bar + (x[t])

x_bar = x_bar / N

for m in range(1-1, M):
    for t in range(1-1, N-m):
        c_k[m] = c_k[m] + ((x[t] - x_bar) * (x[t+m] - x_bar))

c_k = (1/N) * c_k

r_k = c_k / c_k[0]

#print(c_k)
#print(c_k.size)

#print(c_k.size)
#print(r_k.size)

# we find the auto-correlation coefficients
# we compute the auto-correlation function, the auto-correlation coefficients, the correlogram

plt.figure()

#plt.plot(r_k, 'bo-', label='auto-correlation coefficients')
plt.plot(range(1,M+1), r_k, 'bo-', label='BTC auto-correlation coefficients')
#plt.plot((1:len(x))*(1/len(x))*21, x, 'bo-', label='Yield')

plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(1/np.sqrt(N)), 'ro--', label='2/sqrt(N)')
plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(-1)*(1/np.sqrt(N)), 'ro--', label='-2/sqrt(N)')

plt.title('The auto-correlation coefficients, the correlogram, for BTC.')
plt.legend()

plt.show()

# we see that the auto-correlation coefficients go to zero

# the auto-correlation coefficients come down quickly towards zero
# from the plot, we observe that the auto-correlation coefficients go to zero fast



# we compute the auto-correlation coefficients for ETH

#x = (?)
#x = array1

#x = ETH_close_ratio
x = ETH_close_ratio2

x = np.asarray(x)

#print(array1.size)
#N = x.size

#N = x.size
N = len(x)
#print(N)

#M = N / 10
#print(M)

import math
M = math.floor(N / 10)
#print(M)

#c_k = np.zeros(int(M))
c_k = np.zeros(M)

x_bar = 0

#print(x[0,0])
#print(x[0,1])

for t in range(1-1,N-1):
    x_bar = x_bar + (x[t])

x_bar = x_bar / N

for m in range(1-1, M):
    for t in range(1-1, N-m):
        c_k[m] = c_k[m] + ((x[t] - x_bar) * (x[t+m] - x_bar))

c_k = (1/N) * c_k

r_k = c_k / c_k[0]

#print(c_k)
#print(c_k.size)

#print(c_k.size)
#print(r_k.size)

# we find the auto-correlation coefficients
# we compute the auto-correlation function, the auto-correlation coefficients, the correlogram

plt.figure()

#plt.plot(r_k, 'bo-', label='auto-correlation coefficients')
plt.plot(range(1,M+1), r_k, 'ro-', label='ETH auto-correlation coefficients')
#plt.plot((1:len(x))*(1/len(x))*21, x, 'bo-', label='Yield')

plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(1/np.sqrt(N)), 'ro--', label='2/sqrt(N)')
plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(-1)*(1/np.sqrt(N)), 'ro--', label='-2/sqrt(N)')

plt.title('The auto-correlation coefficients, the correlogram, for ETH.')
plt.legend()

plt.show()

# we see that the auto-correlation coefficients go to zero

# the auto-correlation coefficients come down quickly towards zero
# from the plot, we observe that the auto-correlation coefficients go to zero fast



# we compute the auto-correlation coefficients for XRP

#x = (?)
#x = array1

#x = XRP_close_ratio
x = XRP_close_ratio2

x = np.asarray(x)

#print(array1.size)
#N = x.size

#N = x.size
N = len(x)
#print(N)

#M = N / 10
#print(M)

import math
M = math.floor(N / 10)
#print(M)

#c_k = np.zeros(int(M))
c_k = np.zeros(M)

x_bar = 0

#print(x[0,0])
#print(x[0,1])

for t in range(1-1,N-1):
    x_bar = x_bar + (x[t])

x_bar = x_bar / N

for m in range(1-1, M):
    for t in range(1-1, N-m):
        c_k[m] = c_k[m] + ((x[t] - x_bar) * (x[t+m] - x_bar))

c_k = (1/N) * c_k

r_k = c_k / c_k[0]

#print(c_k)
#print(c_k.size)

#print(c_k.size)
#print(r_k.size)

# we find the auto-correlation coefficients
# we compute the auto-correlation function, the auto-correlation coefficients, the correlogram

plt.figure()

#plt.plot(r_k, 'bo-', label='auto-correlation coefficients')
plt.plot(range(1,M+1), r_k, 'ko-', label='XRP auto-correlation coefficients')
#plt.plot((1:len(x))*(1/len(x))*21, x, 'bo-', label='Yield')

plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(1/np.sqrt(N)), 'ro--', label='2/sqrt(N)')
plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(-1)*(1/np.sqrt(N)), 'ro--', label='-2/sqrt(N)')

plt.title('The auto-correlation coefficients, the correlogram, for XRP.')
plt.legend()

plt.show()

# we see that the auto-correlation coefficients go to zero

# the auto-correlation coefficients come down quickly towards zero
# from the plot, we observe that the auto-correlation coefficients go to zero fast



# we compute the auto-correlation coefficients for BCH

#x = (?)
#x = array1
x = BCH_close_ratio

x = np.asarray(x)

#print(array1.size)
#N = x.size

#N = x.size
N = len(x)
#print(N)

#M = N / 10
#print(M)

import math

M = math.floor(N / 10)
#print(M)

#c_k = np.zeros(int(M))
c_k = np.zeros(M)

x_bar = 0

#print(x[0,0])
#print(x[0,1])

for t in range(1-1,N-1):
    x_bar = x_bar + (x[t])

x_bar = x_bar / N

for m in range(1-1, M):
    for t in range(1-1, N-m):
        c_k[m] = c_k[m] + ((x[t] - x_bar) * (x[t+m] - x_bar))

c_k = (1/N) * c_k

r_k = c_k / c_k[0]

#print(c_k)
#print(c_k.size)

#print(c_k.size)
#print(r_k.size)

# we find the auto-correlation coefficients
# we compute the auto-correlation function, the auto-correlation coefficients, the correlogram

plt.figure()

#plt.plot(r_k, 'bo-', label='auto-correlation coefficients')
plt.plot(range(1,M+1), r_k, 'mo-', label='BCH auto-correlation coefficients')
#plt.plot((1:len(x))*(1/len(x))*21, x, 'bo-', label='Yield')

plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(1/np.sqrt(N)), 'ro--', label='2/sqrt(N)')
plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(-1)*(1/np.sqrt(N)), 'ro--', label='-2/sqrt(N)')

plt.title('The auto-correlation coefficients, the correlogram, for BCH.')
plt.legend()

plt.show()

# we see that the auto-correlation coefficients go to zero

# the auto-correlation coefficients come down quickly towards zero
# from the plot, we observe that the auto-correlation coefficients go to zero fast



# we have used +/- 2/sqrt(N)

# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 14.3: Examples, 6th edition (2004), Chapman & Hall / CRC.

# we use example 14.2 from Chapter 14.3: Examples
# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 14.3: Examples, 6th edition (2004), Chapman & Hall / CRC.

# all the auto-correlation coefficients go to zero fast and, hence, the time series are stationary
# ARMA(0,1) or ARMA(1,0) can be used






# we compute the auto-correlation coefficients for the one-step difference of BCH

#x = (?)
#x = array1

#x = BCH_close_ratio
#x = BCH_close_ratio[1:] - BCH_close_ratio[0:-1]

x = np.diff(BCH_close_ratio)
# x is the one-step difference

#print(np.size(BCH_close_ratio))
#print(np.size(x))
#print(np.size(BCH_close_ratio[1:]))
#print(np.size(BCH_close_ratio[0:-1]))

x = np.asarray(x)

#print(array1.size)
#N = x.size

#N = x.size
N = len(x)
#print(N)

#M = N / 10
#print(M)

import math
M = math.floor(N / 10)
#print(M)

#c_k = np.zeros(int(M))
c_k = np.zeros(M)

x_bar = 0

#print(x[0,0])
#print(x[0,1])

for t in range(1-1,N-1):
    x_bar = x_bar + (x[t])

x_bar = x_bar / N

for m in range(1-1, M):
    for t in range(1-1, N-m):
        c_k[m] = c_k[m] + ((x[t] - x_bar) * (x[t+m] - x_bar))

c_k = (1/N) * c_k

r_k = c_k / c_k[0]

#print(c_k)
#print(c_k.size)

#print(c_k.size)
#print(r_k.size)

# we find the auto-correlation coefficients
# we compute the auto-correlation function, the auto-correlation coefficients, the correlogram

plt.figure()

#plt.plot(r_k, 'bo-', label='auto-correlation coefficients')
plt.plot(range(1,M+1), r_k, 'mo-', label='BCH auto-correlation coefficients')
#plt.plot((1:len(x))*(1/len(x))*21, x, 'bo-', label='Yield')

plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(1/np.sqrt(N)), 'ro--', label='2/sqrt(N)')
plt.plot(range(1,M+1), np.ones(np.size(r_k))*2*(-1)*(1/np.sqrt(N)), 'ro--', label='-2/sqrt(N)')

plt.title('The auto-correlation coefficients, the correlogram, for BCH.')
plt.legend()

plt.show()

# we see that the auto-correlation coefficients go to zero

# the auto-correlation coefficients come down quickly towards zero
# from the plot, we observe that the auto-correlation coefficients go to zero fast



from statsmodels.tsa.arima_model import ARIMA
# we fit the ARIMA model

x = BCH_close_ratio

#model = ARIMA(x, order=(1, 0, 0))
#model = ARIMA(x, order=(0, 0, 1))

model = ARIMA(x, order=(1, 1, 0))
#model = ARIMA(x, order=(0, 1, 1))

#model = ARIMA(x, order=(2, 1, 0))
#model = ARIMA(x, order=(0, 1, 2))

model_fit = model.fit(disp=0)
print(model_fit.summary())

# we now use DataFrame
from pandas import DataFrame

# plot residual errors
residuals = DataFrame(model_fit.resid)

residuals.plot()
plt.show()

residuals.plot(kind='kde')
plt.show()

print(residuals.describe())

# we use: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/






# we now use Keras
# we use: https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/

# we use the book of F. Chollet: Deep Learning with Python
# we use: http://amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438/ref=sr_1_1?ie=UTF8&qid=1523486008&sr=8-1&keywords=chollet

import keras

from keras.datasets import mnist

# we use the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(' ')
print(train_images.shape)

print(len(train_labels))
print(train_labels)

print(' ')
print(test_images.shape)

print(len(test_labels))
print(test_labels)



from scipy.io import wavfile

#fs, data = wavfile.read('../Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA1')
fs, data = wavfile.read('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA1')

print('')
print(fs)

print(data.shape)
print(len(data))

from matplotlib import pyplot as plt
import numpy as np

times = np.arange(len(data))/float(fs)

# Make the plot: tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))

plt.fill_between(times, data, color='k')
plt.xlim(times[0], times[-1])

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# we can set the format by changing the extension
# such as .pdf, .svg, .eps
#plt.savefig('plot.png', dpi=100)

plt.show()



# we use the kapre library
# we use: https://github.com/keunwoochoi/kapre
import kapre

#kapre.time_frequency.Spectrogram(n_dft=512, n_hop=256, padding='same', power_spectrogram=2.0, return_decibel_spectrogram=False,
#                                 trainable_kernel=False, image_data_format='default')

# we use kapre
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise

# 6 channels (!), maybe 1-sec audio signal, for an example.
#input_shape = (6, 44100)

sr = len(data)
input_shape = (1, sr)

from keras.models import Sequential
model = Sequential()

# A mel-spectrogram layer
model.add(Melspectrogram(n_dft=512, n_hop=256, input_shape=input_shape,
                         padding='same', sr=sr, n_mels=128,
                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
                         return_decibel_melgram=False, trainable_fb=False,
                         trainable_kernel=False, name='trainable_stft'))

# add some additive white noise
model.add(AdditiveNoise(power=0.2))

# we normalise it per frequency
model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'

# After this, it's just a usual keras workflow. For example..
# Add some layers, e.g., model.add(some convolution layers..)

# Compile the model
#model.compile('adam', 'categorical_crossentropy') # if single-label classification

# train it with raw audio sample inputs
#x = load_x() # e.g., x.shape = (10000, 6, 44100)
#y = load_y() # e.g., y.shape = (10000, 10) if it's 10-class classification

x = data

# and train it
#model.fit(x, y)



# we use: https://github.com/keunwoochoi/kapre/blob/master/examples/example_codes.ipynb

# we use: https://github.com/keunwoochoi/kapre/blob/master/examples/prepare%20audio.ipynb
# we also use: https://github.com/keunwoochoi/kapre/blob/master/examples/example_codes.ipynb

# use pandas
import pandas as pd

import numpy as np

#import keras
#import kapre

import librosa
from librosa import display

y, sr = librosa.load('/Users/dionelisnikolaos/Desktop/folder_desktop/MATLAB_Project2/TIMIT/TRAIN/DR1/FCJF0/wavSA1', sr=None)
#print(len(y), sr)

print(librosa.samples_to_time(len(y), sr))

D = librosa.stft(y)
#print(D.shape, D.dtype)

S, phase = librosa.magphase(D)
#print(S.dtype, phase.dtype, np.allclose(D, S * phase))

plt.figure(figsize=(14, 4))
logPowerSpectrum = np.log(np.abs(librosa.stft(y, 512, 256)) ** 2)

#display.specshow(np.log(np.abs(librosa.stft(y, 512, 256)) ** 2), y_axis='linear', sr=sr)
display.specshow(logPowerSpectrum, y_axis='linear', sr=sr)

plt.title('Log-Spectrogram')
plt.show()

# we use: https://github.com/librosa/tutorial/blob/master/Librosa%20tutorial.ipynb

