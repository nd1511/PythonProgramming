
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

#fig = plt.figure()
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



