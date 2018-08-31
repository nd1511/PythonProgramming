
# we use Chris Chatfield's book
# we use: The Analysis of Time Series: An Introduction, Chris Chatfield, 6th edition (2004), Chapman & Hall / CRC.

# References:
# [1] C. Chatfield, "The Analysis of Time Series: An Introduction" 6th edition (2004), Chapman & Hall / CRC.

# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 2.7: Autocorrelation and the Correlogram, 6th edition (2004), Chapman & Hall / CRC.
# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 14.3: Examples, 6th edition (2004), Chapman & Hall / CRC.

# we use: https://www.crcpress.com/The-Analysis-of-Time-Series-An-Introduction-Sixth-Edition/Chatfield/p/book/9781584883173

# datasets can be found in (https://www.crcpress.com/The-Analysis-of-Time-Series-An-Introduction-Sixth-Edition/Chatfield/p/book/9781584883173)
# C. Chatfield: https://www.crcpress.com/The-Analysis-of-Time-Series-An-Introduction-Sixth-Edition/Chatfield/p/book/9781584883173



# we use: http://www.commsp.ee.ic.ac.uk/~mandic/courses.htm
# we also use: https://pdfs.semanticscholar.org/presentation/27ca/53acde0b5a7914cb042365ae4f05a2c1c0ce.pdf

# http://www.commsp.ee.ic.ac.uk/~mandic/courses.htm
# http://www.econ.ohio-state.edu/dejong/note2.pdf

# for MATLAB, we use: https://github.com/yalamarthibharat/TimeSeriesAnalysis



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



# Chris Chatfield, The Analysis of Time Series: An Introduction, Chapter 14.3: Examples, 6th edition (2004), Chapman & Hall / CRC.

# accordding to example 14.3, (1) c_k, (2) r_k, (3) one-step difference and repeat (1) and (2)
# we have done (1) and (2) and now we do (3)

#x = (?)
x = array1[0,1:] - array1[0,0:-1]
# x is the one-step difference

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

# we create a dataframe
#df = pd.read_csv('./dataset_cryptocurrency.csv')
df = pd.read_csv('/Users/dionelisnikolaos/Downloads/dataset_cryptocurrency.csv')

# output first 5 rows in dataset
print(df.head())






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






