
import numpy as np

import torch

from torch.autograd import Variable

import matplotlib.pyplot as plt

import pandas as pd



# we use pandas dataframes

# we import our dataset into a pandas dataframe
df = pd.read_csv('Iris.csv')

df[['Species']] = df['Species'].map({'Iris-setosa':0, 'Iris-virginica':1, 'Iris-versicolor':2}) #map text labels to numberical vaules

# we shuffle our dataset
df = df.sample(frac=1)
# this is important for training

# we convert our data into torch tensors
X = torch.Tensor(np.array(df[df.columns[1:-1]])) #pick our features from our dataset
Y = torch.LongTensor(np.array(df[['Species']]).squeeze()) #select our label - squeeze() removes redundant dimensions

#size of training set
m = 100

#split our data into training and test set

#training set
x_train = Variable(X[0:m])
y_train = Variable(Y[0:m])

#test set
x_test = Variable(X[m:])
y_test = Variable(Y[m:])



#define model class - inherit useful functions and attributes from torch.nn.Module
class logisticmodel(torch.nn.Module):
    def __init__(self):
        super().__init__() #call parent class initializer
        self.linear = torch.nn.Linear(4, 3) #define linear combination function with 4 inputs and 3 outputs

    def forward(self, x):
        pred = self.linear(x) #linearly combine our inputs to give 3 outputs
        pred = torch.nn.functional.softmax(pred, dim=1) #activate our output neurons to give probabilities of belonging to each of the three class
        return pred

#training hyper-parameters
no_epochs = 100

lr = 0.1



#create our model from defined class
mymodel = logisticmodel()
costf = torch.nn.CrossEntropyLoss() #cross entropy cost function as it is a classification problem
optimizer = torch.optim.Adam(mymodel.parameters(), lr = lr) #define our optimizer

#for plotting costs
costs=[]
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cost')
ax.set_xlim(0, no_epochs)
plt.show()

#training loop - same as last time
for epoch in range(no_epochs):
    h = mymodel.forward(x_train) #forward propagate - calulate our hypothesis

    #calculate, plot and print cost
    cost = costf(h, y_train)
    costs.append(cost.data[0])
    ax.plot(costs, 'b')
    fig.canvas.draw()
    print('Epoch ', epoch, ' Cost: ', cost.data[0])

    #calculate gradients + update weights using gradient descent step with our optimizer
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    #Some people's laptops are too fast so the plot updates too fast to be visible - uncomment the following line to fix that problem
    #plt.pause(0.0001)

#test accuracy
test_h = mymodel.forward(x_test) #predict probabilities for test set
_, test_h = test_h.data.max(1) #returns the output which had the highest probability
test_y = y_test.data
correct = torch.eq(test_h, test_y) #perform element-wise equality operation
accuracy = torch.sum(correct)/correct.shape[0] #calculate accuracy
print('Test accuracy: ', accuracy)



#predict the class of an input using our trained model
inp = [4.6, 3.1, 1.2, 0.3] #define our inputs
inp = Variable(torch.Tensor(inp)) #convert our input to variable
prediction = mymodel.forward(inp) #calculate our output probabilities
_, prediction = prediction.data.max(1) #which class had the highest probability

print(prediction)





