
#import torchvision

import numpy as np

import torch

from torch.autograd import Variable

import matplotlib.pyplot as plt

import pandas as pd

#df = pd.read_csv('/Users/dionelisnikolaos/Downloads/cancer_data.csv')

df = pd.read_csv('/Users/dionelisnikolaos/Downloads/Iris.csv')

# we use "df[['Species']]" to change the data in the data frame
# we use "df['Species']" to access the data in the data frame

df[['Species']] = df['Species'].map({'Iris-setosa':0, 'Iris-virginica':1, 'Iris-versicolor':2})

#print(df.head(5))

# we shuffle the dataset, we do not want an ordered dataset
df = df.sample(frac=1)

#print(df.head())

print(len(df))

# we define the features X
X = torch.Tensor(np.array(df[df.columns[1:-1]]))

# the LongTensor can only hold integer values
Y = torch.LongTensor(np.array(df[['Species']]).squeeze())
# we reduce the dimensions using ".squeeze()"

# we now separate training and test set
# we create a training set and a test set

# m is the size of the training set
m = 100

# we now separate training and test set

# we create the training set
x_train = Variable(X[0:m])
y_train = Variable(Y[0:m])

# we create the test set
x_test = Variable(X[m:])
y_test = Variable(Y[m:])



# we define the model class
class logisticmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4,3)

    def forward(self, x):
        pred = self.linear(x)

        # we use the softmax operation
        pred = torch.nn.functional.softmax(pred, dim=1)
        # softmax is the normalized exponential operation

        # the softmax output layer is used for multi-class classification
        # the outputs are probabilities

        return pred



# we define training hyperparameters

# we define the number of epochs
no_epochs = 100

# we define the learning rate, the step-size
lr = 0.1

# create our model from defined class
mymodel = logisticmodel()

# we create the loss/cost function
costf = torch.nn.CrossEntropyLoss()

# we use the cross-entropy CE cost function
# we use "torch.nn.CrossEntropyLoss()" for the CE loss function

# we use Adam
optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)



# for plotting cost
costs = []

plt.ion()

fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_xlabel("Epoch")
ax.set_ylabel("Cost")

ax.set_xlim(0, no_epochs)

plt.show()



for epoch in range(no_epochs):
    h = mymodel.forward(x_train)
    cost = costf(h, y_train)

    cost = costf(h, y_train)

    costs.append(cost.data[0])

    ax.plot(costs, 'b')

    fig.canvas.draw()

    # cost is a variable and we use "cost.data[0]"
    print("Epoch: ", epoch, "Cost: ", cost.data[0])

    # we compute gradients based on previous gradients (i.e. momentum)

    # we use momentum and we set the gradients to zero
    optimizer.zero_grad()

    # we backpropagate the cost
    cost.backward()

    optimizer.step()

    # we pause the plot so as to see the graph
    plt.pause(0.001)



# we now compute the accuracy
test_h = mymodel.forward(x_test)

# we use argmax, we choose the class with the highest probability

# values, ind = test_h.data.max(1)
_, test_h = test_h.data.max(1)

test_y = y_test.data

correct = torch.eq(test_h, test_y)

print(test_h[:5])
print(test_y[:5])

print(correct[:5])

# we compute the accuracy of our model
accuracy = torch.sum(correct) / correct.shape[0]

print('Test accuracy: ', accuracy)



# we run the training and testing procedure many times
# and we keep the model with the best accuracy

# we run this many times and, in the end, we keep the model with the best test results



# we now make predictions with our model
#inp = Variable(torch.Tensor([4, 3.7, 1, 0.5]))

# we compute the probabilities
#prediction = mymodel.forward(inp)

# we print the probabilities
#print(prediction)
# we choose the class with the highest probability

# we then use argmax, we choose the class with the highest probability



