
import torch

from torch.autograd import Variable

import torchvision

from torchvision import transforms, datasets

import matplotlib.pyplot as plt

# we import several useful functions
import torch.nn.functional as F



# we use SGD, stochastic gradient descent
# we use mini-batches and SGD

# we define the size of the mini-batches
batch_size = 256

# we define the learning rate, the step size
lr = 0.5

epochs = 1



# we use the MNIST database
# we use MNIST and the hand-written digits database

training_data = datasets.MNIST(root = 'data/',
                               transform = transforms.ToTensor(),
                               train = True,
                               download=True)

print(training_data[0])

plt.imshow(training_data[0][0][0])

#plt.show()



test_data = datasets.MNIST(root = 'data/',
                           train = False,
                           transform = transforms.ToTensor())

# we create the training samples
training_samples = torch.utils.data.DataLoader(dataset = training_data,
                                               batch_size = batch_size,
                                               shuffle = True)

# we create the test samples
test_samples = torch.utils.data.DataLoader(dataset = test_data,
                                           batch_size = batch_size,
                                           shuffle = False)
# we do not need to shuffle because this is testing



print("Number of training examples: ", len(training_samples.dataset))

print("Number of test examples: ", len(test_samples.dataset))



class convnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # we now define the layers
        # kernel = filter = convolution

        # 1 input channel, 5 output channels
        self.conv1 = torch.nn.Conv2d(1, 5,
                                     kernel_size = 5)

        # 5 input channel, 10 output channels
        self.conv2 = torch.nn.Conv2d(5, 10,
                                     kernel_size = 3)

        # we use ReLU
        self.relu = torch.nn.ReLU()

        # we now define the dense fully-connected NN
        self.dense = torch.nn.Linear(4840, 10)

    def forward(self, x):

        # we use conv1
        out1 = self.relu(self.conv1(x))

        # we use conv2
        out2 = self.relu(self.conv2(out1))

        print(x.size(0))

        # we unwrap this to a string, to a column
        todense = out2.view(x.size(0), -1)

        output = self.dense(todense)

        #return output
        return F.log_softmax(output, dim = 0)
        # we use log softmax along the zeroth dimension



mynet = convnet()

# we use the negative log likelihood (NLL)
criterion = torch.nn.NLLLoss()

# NLL is like cross-entropy CE, when one-hot coding
# we use one-hot vectors and we use NLL

# we use SGD
optimiser = torch.optim.SGD(mynet.parameters(), lr = lr)



def train():
    # we now set our model ready for training
    mynet.train()

    # dropout changes training and testing conditions

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.grid()
    plt.ion()

    plt.show()

    ax.set_xlabel('Batch')

    ax.set_ylabel("Loss")

    costs = []

    for e in range(epochs):

        # we use mini-batches
        for i, (features, labels) in enumerate(training_samples):
            features, labels = Variable(features), Variable(labels)

            prediction = mynet(features)

            optimiser.zero_grad()

            # we use NLL
            loss = criterion(prediction, labels)

            # one-hot coding is automatically done

            loss.backward()

            optimiser.step()



            costs.append(loss.data)

            ax.plot(costs, 'b')

            fig.canvas.draw()

            # we pause the plot so as to see the graph
            plt.pause(0.001)



            # loss is a variable and we use "loss.data[0]"
            print("Epoch: ", e, "Batch: ", i, "Loss: ", loss.data[0])

            # we stop training at 50th batch
            if i == 50:
                break



train()



def test():

    print('\n\n\n')

    # we use "eval()"
    mynet.eval()

    correct = 0

    for features, labels in test_samples:
        features, labels = Variable(features), Variable(labels)

        probabilities = mynet(features)

        prediction = probabilities.data.max(1)[1]

        # we use "eq()" to have the same format
        correct += (prediction.eq(labels.data.view_as(prediction))).sum()

    print("Test set accuracy: ", correct / len(test_samples.dataset))



test()




