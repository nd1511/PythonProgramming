
# we use pandas for dataframes
import pandas as pd
# we use data-frames

import numpy  as np

import torch

import torch.nn.functional as F

from torch.autograd import Variable

import matplotlib.pyplot as plt



#df = pd.read_csv('/Users/dionelisnikolaos/Downloads/cancer_data.csv')
#df = pd.read_csv('/Users/dionelisnikolaos/Downloads/Iris.csv')

df = pd.read_csv('/Users/dionelisnikolaos/Downloads/creditcard.csv')

# we see the data
#print(df)

# the data are transactions, is the transaction fraudulent?

# the data have been transformed with PCA, Principal Component Analysis

# the data have been transformed with PCA
# V1 is the most important

n = df.shape[1]
m = df.shape[0]

#print(n,m)
# 31 284807

# here, n = 31 and m = 284807



alldata = np.array(df)

# we make a tuple

# we make a tuple of features and labels
alldata = (alldata[:,:5], alldata[:,-1])
# we keep only the first 5 columns, we keep the first 5 from PCA

# we have an imbalanced data set
# most data are good, most transactions are legit and nont fraudulent

frauds = np.array(df[df['Class'] == 1])

num_frauds = frauds.shape[0]
print(num_frauds)

legit = np.array(df[df['Class'] == 0])

# we use np.random.randint(), we use random integers
traininglegits = np.random.randint(m - num_frauds, size=num_frauds)
# we create a balanced training set

# we now have a balanced training set

trainingdata = np.vstack((frauds, legit[traininglegits]))

# we use a tuple
trainingdata = (trainingdata[:,:5], trainingdata[:,-1])

# this is the end of data pre-processing



lr = 0.01

epochs = 50

class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # linear layer 1
        self.l1 = torch.nn.Linear(5, 6)

        # linear layer 2
        self.l2 = torch.nn.Linear(6,3)

        # linear layer 3
        self.l3 = torch.nn.Linear(3,1)
        # the output is 0 or 1, is it legit or fraud?

        self.relu = F.relu

        self.sigmoid = F.sigmoid

    def forward(self, x):
        out1 = self.relu(self.l1(x))

        out2 = self.relu(self.l2(out1))

        out3 = self.sigmoid(self.l3(out2))

        return out3



# we now create mynet
mynet = NN()

# we use the MSE cost function
criterion = torch.nn.MSELoss()

# we use the CE cost function, we use cross-entropy
#criterion = torch.nn.NLLLoss()
# we use negative log-likelihood, we use NLL

# we use Rprop
#optimiser = torch.optim.Rprop(mynet.parameters(), lr=lr)

# we use Adam
optimiser = torch.optim.Adam(mynet.parameters(), lr=lr)



def train():
    mynet.train()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # we turn on the interactive mode
    plt.ion()

    plt.show()

    costs = []

    for e in range(epochs):

        features, labels = Variable(torch.Tensor(trainingdata[0])), Variable(torch.Tensor(trainingdata[1]))

        # we now use forward propagation
        prediction = mynet(features)

        # we set gradients to zero
        optimiser.zero_grad()

        loss = criterion(prediction, labels)

        loss.backward()

        optimiser.step()

        # we append our list costs
        costs.append(loss.data)

        ax.plot(costs, 'b')

        fig.canvas.draw()

        # we pause the plot so as to see the graph
        plt.pause(0.001)

        print('Epoch', e, '\tLoss', loss.data[0])



train()



def test():
    print('\n\n\n')

    test_size = 2048

    test_sample = np.random.randint(m, size=test_size)

    features, labels = Variable(torch.Tensor(alldata[0][test_sample])), Variable(torch.Tensor(alldata[1][test_sample]))

    # we used torch.Tensor(alldata[0][test_sample])
    # we used torch.Tensor(alldata[1][test_sample])

    prediction = np.round(mynet(features).data)

    # we output a binary vector
    #correct = prediction.eq(labels.data.view_as(prediction))

    correct = prediction.eq(labels.data.view_as(prediction)).sum()

    print('Test accuracy', correct/test_size)



test()



