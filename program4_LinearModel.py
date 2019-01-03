import numpy as np

"""
comprehensions
list comprehensions

A list comprehension is created using [i for i in list1 if i%2 == 0].
The output of a list comprehension is a new list.

The syntax is: result = [transform iteration filter].

filter => filtering condition
The transform occurs for every iteration if the filtering condition is met.
"""

# list comprehensions: one-line code

# The syntax is: result = [transform iteration filter].
# The order in the list matters and the new list has this order.

lista = [1,2,3,4,5,6,7,8,9]
print([i*2 for i in lista])

lista.pop()
print(lista)

lista.pop(0)
print(lista)
print('')

# print all the multiples of 3 and 7 from 1 to 100 using list comprehension
list1 = [i for i in range(1,101) if i%3 == 0 and i%7 == 0]
print(list1)

# 2D array 6x6 with zeros
array2D = [[0 for i in range(0,6)] for j in range(0,6)]
print(array2D)

array2D[0][0] = 1
print(array2D)
print('')

# 3D array 6x6 with zeros
array3D = [[[0 for i in range(6)] for j in range(6)] for k in range(6)]
print(array3D)

print(array3D[0][0][0])
print('')



"""
dictionary

dict = HashMap
dictionaries have keys and values
"""

# =============================================================================
# # Create a function that adds a specific value to the value of a key
# # and if the key does not exist, then create the key.
# =============================================================================

def function1(dict1, key1, value1):
    if key1 not in dict1:
        dict1[key1] = 0

    dict1[key1] += value1
    return dict1

def function2(dict1, key1, value1):
    dict1[key1] = dict1.get(key1, 0) + value1
    return dict1

d1 = {
    'milk': 3.67,
    'butter': 1.95,
    'bread': 1.67,
    'cheese': 4.67
}
print(d1)

d1['butter'] = 2.35
print(d1)
print('')

d2 = {
    1: 3.67,
    2: 1.95,
    3: 1.67,
    4: 4.67
}
print(d2)

d2[2] = 2.35
print(d2)
print('')

d3 = dict([('milk', 3.76), ('butter', 1.95), ('bread', 1.67), ('cheese', 4.67)])
print(d3)

del d3['butter']
print(d3)
print('')

# we use ".format(.)"
print('length of dictionary d3 = {} '.format(len(d3)))

print('length of dictionary d3 = {} compared to {} i in {} '.format(len(d3), d1, d2))
print('')

print(d3.keys())
print(d3.values())

print(d3.items())
print('')

# list1 = dict1.items()
# ".items()" returns a list of tuples

# traverse a dictionary
for food in d1:
    print('{} costs {}'.format(food, d1[food]))

print('')
d1 = function1(d1, 'whine', 4.15)

d1 = function1(d1, 'milk', 1)
print(d1)

d1 = function2(d1, 'whine2', 3.15)
d1 = function2(d1, 'milk', 1)

print(d1)
print('')

# use comprehensions

# use dict comprehension
d4 = {k: v for k, v in enumerate('Good Year John')}
print(d4)

# dict with all letters in "Good Year John"
# without the letters in "John"

d5 = {k: v for k, v in enumerate("Good Year John") if v not in "John"}
print(d5)
print('')

# dict comprehensions => one-line code

# list1 = dict1.keys()
# ".keys()" returns a list

# list2 = dict1.values()
# ".values()" returns a list

# list3 = dict1.items()
# ".items()" returns a list of tuples



"""
Sets
A set has no dublicates.
"""

s = {'a','b','a','c','d'}
print(s)

s2 = set("Good Year John")
print(s2)
print('')

a = set('12345678a')
b = set('1234b')

print('A = ',a)
print('B = ',b)
print('')

a.add('9')
b.remove('4')

print('A = ',a)
print('B = ',b)
print('')

print('A - B = ',a-b) #difference
print('A | B = ',a|b) #Union

print('A & B = ',a&b) #intersection
print('A ^ B = ',a^b) #symmetric difference
print('')



"""
Graphs
Use dict to create graphs.

Graphs are dictionaries in Python.
Dictionaries have keys and values, where the key is the index.

Graphs solve maze problems.
We have directed and undirected graphs.
"""

# traverse a graph
# graphs: binary graphs are a special case of graphs

# maze => graphs
# graphs solve maze problem

# we use a dictionary to create a graph

# graphs are dictionaries
# use dictionaries, lists and sets

# depth first search (dfs)
def dfs(graph, start):
    visited, stack = set(), [start]

    while stack:
        vertex = stack.pop()

        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)

    return visited

# do depth first search (dfs)
def dfs_paths(graph, start, goal):
    stack = [(start, [start])]

    while stack:
        (vertex, path) = stack.pop()

        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

# breadth first search (bfs)
def bfs(graph, start):
    '''
    help bfs
    '''
    visited, queue = set(), [start]

    while queue:
        vertex = queue.pop(0)

        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)

    return visited

# do breadth first search (bfs)
def bfs_paths(graph, start, goal):
    queue = [(start, [start])]

    while queue:
        (vertex, path) = queue.pop(0)

        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

# crate a graph using a dictionary
graph1 = {'A': set(['B', 'C']),
          'B': set(['A', 'D', 'E']),
          'C': set(['A', 'F']),
          'D': set(['D']),
          'E': set(['B', 'F']),
          'F': set(['C', 'E'])}

# hashmap = dict
# dictionaries are hashmaps

# use: help(dict)
# we use: help(dict) and help (list)

# dict: key and value and index = key

print(dfs(graph1, 'A'))
print(list(dfs_paths(graph1, 'C', 'F')))

print('')
print(bfs(graph1, 'A'))

print(list(bfs_paths(graph1, 'C', 'F')))
print(list(bfs_paths(graph1, 'A', 'F')))

# pandas use dictionaries with list in key and with list in value

# dictionaries have keys and values
# pandas => list in both key and value

# help(dict)

# use: dict(.), set(.), list(.)
# we use: len(dict1)

# from list to dict: dict(list1)
# dict(list1) where list1 has tuples, list1 is a list of tuples

# for OOP, we use classes
# define classes for OOP in Python



# define a list
list_ls = [6, 7, 8, 10]
print('')

# print the firsr item in the list
print(list_ls[0])

# print the last item in the list
print(list_ls[-1])

# print the first, the second and the third items
print(list_ls[0:3])

# create a 2D list
vec1 = [1, 2, 3]
vec2 = [3, 4, 5]

list_ls2 = [vec1, vec2]

print(list_ls2[0])
print(list_ls2[0][0])

print(list_ls2[0][0:2])

# the list can hold different data types
list_ls3 = [4, 4.0, "name1"]

print(list_ls3[0])

print(list_ls3[2])

# print the last item
print(list_ls3[-1])



my_dict = {1:"John", 2:"Mike", 3:"Nick"}

print(my_dict[1])

my_dict2 = {"001":"John", "002":"Mike", "003":"Nick"}

print(my_dict2["001"])

print(my_dict2["003"])

my_boolean = False

print(my_boolean or True)

print(my_boolean and True)

x = 5
y = 10

if x>=y:
    print("OK")
else:
    print("Not OK")

if x==4:
    print("main 1")
elif x!=7:
    print("main 2")
else:
    print("main 3")

f = 0
while f<10:
    print(f)
    f += 2

# n is from 0 to 4
for n in range(5):
    print(n)

my_list = ["this", "is", "a", "list"]

for n in my_list:
    print(n)



# define class and object

# we define the car object
class car:
    # we now initialize the car object
    def __init__(self, brand, year):
        self.brand = brand
        self.year = year
        # self refers to this object
        # self is important, self refers to this car object

# car is an object
# the attributes of the object are brand and year
# we have defined two attributes for our object

my_car = car("ford", 1980)

print(my_car.year)
print(my_car.brand)

my_second_car = car("tesla", 2016)

print(my_car.year)
print(my_second_car.year)



# we define the object car
class car:
    def __init__(self, brand, year):
        self.brand = brand
        self.year = year
        self.state = "parked"
        self.stopping = "not stopping"
    # we define a function for this object
    def start(self):
        if self.year<=1990:
            self.state = "stalling"
        else:
            self.state = "going"
        #print(self.state)
    def stop(self, str1):
        self.stopping = str1

mycar = car("ford", 1980)
#mycar.start()

print(mycar.stopping)

mycar.stop("stopping")
print(mycar.stopping)



print(mycar.state)

#print(mycar.brand)

mycar.start()
print(mycar.state)

mycar2 = car("tesla", 2016)
#mycar2.start()

print(mycar2.state)

mycar2.start()
print(mycar2.state)

#print(mycar.start())
#print(mycar2.start())

#string1 = mycar2.start()

#print(string1)

#vec = numpy.array([[1,2,3], [4,5,6]])

#import numpy
import numpy as np

vec = np.array([[1,2,3], [4,5,6]])

print(vec)

matrix1 = np.array([[1,2,3], [4,5,6]])

vec1 = np.array([0, 0, 1])

vec2 = np.matmul(matrix1, vec1)

print(vec2)



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



# import functionality from these libraries

# for efficient numerical computation
import numpy as np

# for building computational graphs
import torch
# computational graphs are computational networks

# for automatically computing gradients of our cost with respect to what we want to optimise
from torch.autograd import \
    Variable

# for plotting absolutely anything
import matplotlib.pyplot as plt

# for plotting 3D graphs
from mpl_toolkits.mplot3d import Axes3D

# define hyper-parameters as needed
n = 2  # number of features e.g. number of windows, number of rooms
m = 50  # number of training examples, e.g. the number of windows and rooms was measured for m houses

epochs = 100  # how many times do we want to run through the data to train our model
lr = 0.05  # what proportion of our gradient do we want to update our parameters by

# create dataset and variables - built in models use one row per data point rather than one column
X = Variable(torch.rand(m, n))
Y = Variable(2 * X.data[:, 0] + 1.6 * X.data[:, 1] + 1)

plt.ion()

# create a figure
fig = plt.figure(figsize=(10, 10))

# create our first axes to plot our data in space
ax1 = fig.add_subplot(121, projection='3d')  # start with 111
x1 = np.arange(2)  # meshgrid takes two vectors which would be a range of numbers along each axis
x2 = np.arange(2)  # it outputs two matrices where a pair of the same position element gives a coordinate
x1, x2 = np.meshgrid(x1, x2)  # covering the entire domain of the input plane
ax1.scatter(X.data[:, 0], X.data[:, 1], Y.data[:])  # plot data points
ax1.set_xlabel('Windows')
ax1.set_ylabel('Rooms')
ax1.set_zlabel('House Price')

# add another set of axes to plot our cost decreasing with iteration
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')
ax2.set_xlim(0, epochs)
ax2.grid()
plt.show()



# define model class
class LinearModel(torch.nn.Module):  # import functionality from a torch module
    def __init__(self):
        super().__init__()  #
        self.linear = torch.nn.Linear(2, 1)  # create a linear layer with 2 inputs leading to 1 output

    def forward(self, x):  # define computation for data passing through model
        y_pred = self.linear(x)  # prediction is a linear combination of inputs
        return y_pred


# create model and define training params
m = LinearModel()

criterion = torch.nn.MSELoss(size_average=True)  # our loss function is the mean squared error
optimizer = torch.optim.SGD(m.parameters(), lr=lr)  # our optimisation technique is stochastic gradient descent

# to keep history of costs to plot against time
costs = []

# train model
for epoch in range(epochs):
    y_pred = m(X)

    cost = criterion(y_pred, Y)  # calculate our current cost

    costs.append(cost.data)  # add current cost to the list of cost histories
    print('Epoch ', epoch, ' Cost: ', cost.data[0])

    # calling J.backwards() does not set the values in the var.grad,it adds to them
    # otherwise it will contain the cumulative history of grads
    optimizer.zero_grad()

    # find rate of change of J with respect to each rg=True variable and put that in tht var.grad
    cost.backward()

    # update the model by moving each parameter in a the direction that most reduces error
    optimizer.step()

    # the bias and weights of the model are generated by the framework of the model
    # get the current values of bias and weights and set them as variables
    w, b = [m.state_dict()[i][0] for i in m.state_dict()]

    # print our parameters
    print('w1', w[0], ' \tw2', w[1], '\tb', b)

    # plot costs
    ax2.plot(costs, 'b')

    y = b + x1 * w[0] + x2 * w[1]  # calculate hypothesis surface

    s = ax1.plot_surface(x1, x2, y, color=(0, 1, 1, 0.5))  # plot surface hypothesis
    ax1.view_init(azim=epoch)  # choose view angle of 3d plot (make it rotate)

    fig.canvas.draw()  # draw the new stuff on the canvas
    s.remove()  # remove the 3d surface plot object

# use model to predict new value
v = Variable(torch.Tensor([[4, 0]]))  # the tensore needs to be made a variable
print('Predict [4, 0]: ', m.forward(v).data[0][0])  # put v forward through the model

# save variables for which rg =True
print(m.state_dict())
torch.save(m.state_dict(), 'saved_linear_model')

# load model
# m = Model()
# m.load_state_dict(torch.load('savedmodel'))

