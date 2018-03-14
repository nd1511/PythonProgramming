
import numpy as np
import torch

from torch.autograd import Variable

import matplotlib.pyplot as plt

import pandas as pd

#df = pd.read_csv('cancer_data.csv')
#print(df.head)

#/Users/dionelisnikolaos/Downloads
df = pd.read_csv('/Users/dionelisnikolaos/Downloads/cancer_data.csv')

# binary classification, M is malign, B is benign
df[['diagnosis']] = df['diagnosis'].map({"M":0, "B":1})

# we shuffle the dataset
df = df.sample(frac=1)

X = torch.Tensor(np.array(df[df.columns[2:-1]]))
Y = torch.Tensor(np.array(df[['diagnosis']]))

m = 450

# splitting into training set and test set
x_train = Variable(X[:m])
y_train = Variable(Y[:m])

x_test = Variable(X[m:])
y_test = Variable(Y[m:])



# create the model class
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # torch.nn.linear(.) performs a linear operation
        self.h1 = torch.nn.Linear(30, 10)
        # the inputs to the NN are 30, we have 30 features

        # the hidden layer has 10 neurons

        self.out = torch.nn.Linear(10, 1)

    def forward(self, x):
        # this is the linear combination
        h1 = self.h1(x)

        # this is the activation, this is the nonlinearity

        # this is the ReLU nonlinearity
        h1 = torch.nn.functional.relu(h1)

        out = self.out(h1)

        # we now define the output layer, the final layers is a sigmoid layer
        out = torch.nn.functional.sigmoid(out)

        return out



# create model object from class
mynet = Net()

# we use the MSE error
criterion = torch.nn.MSELoss()

# training hyperparameters
#no_epochs = 150

no_epochs = 100

#lr = 0.1
lr = 0.003

# Rprop(.) is the optimizer that we use
#optimizer = torch.optim.Rprop()

# Rprop(.) is the optimizer that we use
optimizer = torch.optim.Rprop(mynet.parameters(), lr=lr)
# we use the learning rate lr

# we use Rprop(.), but we could have used RMSprop(.)
# RMSprop(.) is different from Rprop(.)

# we use Rprop(.), but we could have used SGD(.)
# SGD(.) is different from Rprop(.)



# we plot the costs
costs = []
plt.ion()

# we now create a figure
fig = plt.figure()

# 111 means 1 height, 1 wide, and this is the first figure
ax = fig.add_subplot(111)

ax.set_xlabel("Epoch")
ax.set_ylabel("Cost")

plt.show()



# we now train the model
for epoch in range(no_epochs):
    # we forward propagate
    h = mynet.forward(x_train)

    # we calulcate our cost, we use the MSE cost
    cost = criterion(h, y_train)

    # we backpropagate, we use gradient descent (SGD)
    optimizer.zero_grad()
    cost.backward()
    # the cost.backward is based on the computational graph

    # we use an adaptive learning rate
    optimizer.step()
    # we use a step-size, learning rate chosen by the program

    print("Epoch:", epoch, ", Cost:", cost.data[0])
    # "cost" is a variable and we use cost.data[0]

    costs.append(cost.data[0])

    ax.plot(costs, 'b')

    fig.canvas.draw()

    #plt.pause(1)
    plt.pause(0.1)



test_h = mynet.forward(x_test)

# we set the output to be 0 or 1

# we have a binary output, we set the output to be 0 or 1 only
test_h.data.round_()

correct = test_h.data.eq(y_test.data)

# we now compute the accuracy

# we divide by the length, we divide by "correct.shape[0]"
accuracy = torch.sum(correct)/correct.shape[0]

print(accuracy)



# we store our parameters

# we use the state dictionary "mynet.state_dict()"
#torch.save(mynet.state_dict(), "mynet_trained")

#mynet.load_state_dict(torch.load("mynet_trained"))

# we can run the model many times and choose the best parameters
# we can choose the best parameters, the parameters that give the best accuracy






