
import torch
import numpy as np

import matplotlib.pyplot as plt

from torch.autograd import Variable

def makedata(numdatapoints):
    x = np.linspace(-10, 10, numdatapoints)

    #coeffs = [0.5, 5]
    # here, 5 is the bias, it is 5*x^0=5

    coeffs = [2, 0.5, 5]

    # polynomial, we evaluate a polynomial
    y = np.polyval(coeffs, x)
    # polyval is used to create a polynomial dataset

    # we now add noise, we add additive noise
    y += 2 * np.random.rand(numdatapoints)
    # rand(.) is for numbers 0 to 1

    return x,y

numdatapoints = 10

# we now create the data
inputs, labels = makedata(numdatapoints)
# we create a labeled dataset

fig = plt.figure(figsize=(10, 20))

ax1 = fig.add_subplot(121)
# 121 means 1 height, 2 wide, and this is the first figure

ax1.set_xlabel("Input")
ax1.set_ylabel("Output")

#ax1.scatter(np.array(inputs), np.array(labels), s=5)
# the "s=5" is the size of the data points

# we create a scatter plot, we plot y against x
ax1.scatter(np.array(inputs), np.array(labels), s=8)

ax1.grid()

ax2 = fig.add_subplot(122)

ax2.set_title("Error vs Epoch")
ax2.grid()

line1, = ax1.plot(inputs, inputs)
# here, we do not care about the second output

# ion(.) is interactive on

# ion(.) is interactive on, we will update the graphs interactively
plt.ion()

plt.show()



def makefeatures(power):
    features = np.ones((inputs.shape[0], len(powers)))
    # len(powers) = number of columns

    for i in range(len(powers)):
        features[:,i] = (inputs**powers[i])

    print(features.shape)

    return features.T



class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # we create a linear layer in our model
        self.l = torch.nn.Linear(features.shape[0], 1)
        # the inputs is features.shape[0]
        # the output is 1

    def forward(self, x):
        out = self.l(x)
        return out

# the list of hyperparameters
epochs = 50

# lr is the learning rate
#lr = 0.2

#lr = 0.000003
lr = 0.000003

#powers = [1, 2]
powers = [1, 2, 3]

# we now create the features
features = makefeatures(powers)

# features.T means transpose of features

datain = Variable(torch.Tensor(features.T))
# we use transpose .T

labels = Variable(torch.Tensor(labels.T))
# we use transpose .T

# we now create our model
mymodel = LinearModel()

# we now use the MSE cost function
criterion = torch.nn.MSELoss(size_average=True)
# size_average=True means divide my m, where m is the number of training data

#criterion = torch.nn.MSELoss()

# we use stochastic gradient descent (SGD)
optimiser = torch.optim.SGD(mymodel.parameters(), lr=lr)
#lr=lr defines the learning rate, the step size of SGD



def train():
    costs = []

    for e in range(epochs):
        prediction = mymodel(datain)

        # our criteron is the MSE
        cost = criterion(prediction, labels)

        # we now use append(.), list1.append(.)
        costs.append(cost.data)

        print("Epoch", e, "Cost", cost.data[0])

        # we get our parameters out
        params = [mymodel.state_dict()[i][0] for i in mymodel.state_dict()]

        # we set our parameters equal to a list that we define
        # we use i, and when i = 1 then we get the first elements out of the dictionary

        weights = params[0]
        bias = params[1]

        optimiser.zero_grad()

        # we propagate the errors back
        cost.backward()
        # we propagate the errors back using gradients, derivatives

        optimiser.step()

        # we now define the line "line1"

        # torch.mm is matrix multiplication
        line1.set_ydata(torch.mm(weights.view(1,-1), datain.data.t()) + bias)
        # we add a bias term to the matrix-vector multiplication (i.e. to the inner product)

        fig.canvas.draw()
        ax2.plot(costs)

        plt.pause(1)


train()




