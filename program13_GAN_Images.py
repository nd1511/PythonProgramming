# we use a GAN

# use PytTorch
import torch

# use autograd
from torch.autograd import Variable

import torchvision
from torchvision import transforms, datasets

import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# GAN, Generative Adversarial Network
# we use GANs with images

# we define the batch size

#batch_size = 200
batch_size = 100

# import the datasets
train_data = datasets.FashionMNIST(root='fashiondata/',
                                   transform=transforms.ToTensor(),
                                   train=True,
                                   download=True
                                   )

test_data = datasets.FashionMNIST(root='fashiondata/',
                                  transform=transforms.ToTensor(),
                                  train=False,
                                  download=True
                                  )



train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size,
                                            shuffle=True
                                            )
test_samples = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size
                                           )



# we now define the classes

# define the discriminator class
class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)  # 1x28x28-> 64x14x14

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 64x14x14-> 128x7x7

        self.dense1 = torch.nn.Linear(12 8 * 7 *7, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)

        self.bn2 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x))).view(-1, 12 8 * 7 *7)

        x = F.sigmoid(self.dense1(x))

        return x



# define the generator class
class generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(256, 1024)
        self.dense3 = torch.nn.Linear(1024, 12 8 * 7 *7)

        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 128x7x7 -> 64x14x14
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)  # 64x14x14 -> 1x28x28

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.bn3 = torch.nn.BatchNorm1d(12 8 * 7 *7)

        # here, we use "BatchNorm2d(.)"
        self.bn4 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))

        x = F.relu(self.bn3(self.dense3(x))).view(-1, 128, 7, 7)

        x = F.relu(self.bn4(self.uconv1(x)))

        x = F.sigmoid(self.uconv2(x))

        return x



# we now instantiate the model

# we use GPU and ".cuda()"
d = discriminator().cuda()

# we use GPU and ".cuda()"
g = generator().cuda()



# training hyperparameters
no_epochs = 100
dlr = 0.0003
glr = 0.0003

d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)
g_optimizer = torch.optim.Adam(g.parameters(), lr=glr)

dcosts = []
gcosts = []



# training loop
for epoch in range(no_epochs):
    epochdcost = 0
    epochgcost = 0

    # iteratre over mini-batches
    for k, (real_images, _ ) in enumerate(train_samples):
        real_images = Variable(real_images).cuda()

        z = Variable(torch.randn(batch_size, 128)).cuda()
        generated_images = g.forward(z)

        gen_pred = d.forward(generated_images)
        real_pred = d.forward(real_images)

        dcost = -torch.sum(torch.log(real_pred) + torch.log( 1 -gen_pred) ) /batch_size
        gcost = -torch.sum(torch.log(gen_pred) ) /batch_size

        d_optimizer.zero_grad()
        dcost.backward(retain_graph=True)
        d_optimizer.step()

        g_optimizer.zero_grad()
        gcost.backward()
        g_optimizer.step()

        epochdcost += dcost.data[0]
        epochgcost += gcost.data[0]

        if k* batch_size % 10000 == 0:
            g.eval()
            noise_input = Variable(torch.randn(1, 128)).cuda()
            generated_image = g.forward(noise_input)

            plt.figure(figsize=(1, 1))
            plt.imshow(generated_image.data[0][0], cmap='gray_r')
            plt.show()
            g.train()

    epochdcost /= 60000 / batch_size
    epochgcost /= 60000 / batch_size



    print('Epoch: ', epoch)
    print('Disciminator cost: ', epochdcost)
    print('Generator cost: ', epochgcost)

    dcosts.append(epochdcost)
    gcosts.append(epochgcost)



    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.set_xlim(0, no_epochs)

    ax.plot(dcosts, 'b')
    ax.plot(gcosts, 'r')

    fig.canvas.draw()
    plt.show()

# https://drive.google.com/open?id=165pEQUXJUKoP6BZ8-MYyPo_UTRu2n7cB
# use: https://drive.google.com/file/d/1crMeu5xixhM8RchLqU04Px1M8lTwRW80/view?usp=sharing



# we use: https://drive.google.com/open?id=165pEQUXJUKoP6BZ8-MYyPo_UTRu2n7cB
# file: https://drive.google.com/file/d/1crMeu5xixhM8RchLqU04Px1M8lTwRW80/view?usp=sharing

# GAN, Generative Adversarial Network

# we use GANs with images

# In a GAN, a G competes with a D
# G = Generative model and D = Discriminative model

# two models compete, the generator and the discriminator

# the generator creates fake data, generative model
# the discriminator classifies the data as fake or true
# the discriminator has to find the fake data produced by the generator

# minmax game, two models compete in a minimax game
# the one model tries to minimize a function and the other to maximize the function

# min max (E_{x ~ Pdata} {log(D(x))} + E_{Z ~ P(Z)} {log(1-D(G(Z)))})

# min => generator, generative model
# max => discriminator, discriminative model

# Z ~ P(z)
# the D(G(z)) is important, D=Discriminator, G=Generator

# the discriminator distinguishes between true and fake images

# GANs are generative models, they create new data
# generative models compute joint distributions and can create new data

# Generator: min_{\theta_g} (E{Z~P(Z)} {log(1-D(G(Z)))})

# E{log(D(Z))}
# E{.} = mean = average over a large sample, we sample distributions



# two NNs battle each other until they become experts in their own tasks

# GANs are usually used with images
# GANs can synthesize new images

# do GANs find a Nash equilibrium?
# do GANs find a Nash equilibrium in game theory between two opposing tasks?

# use PyTorch
import torch
from torch.autograd import Variable

import torchvision
from torchvision import transforms, datasets

import torch.nn.functional as F

# import libraries
import matplotlib.pyplot as plt

batch_size = 100

train_data = datasets.FashionMNIST(root='fashiondata/',
                                   transform=transforms.ToTensor(),
                                   train=True,
                                   download=True)

test_data = datasets.FashionMNIST(root='fashiondata/',
                                  transform=transforms.ToTensor(),
                                  train=False,
                                  download=True)

# print(train_data[0])
# print(train_data[0][0])

# plt.imshow(train_data[5][0][0], cmap='gray_r')


train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size,
                                            shuffle=True)

test_samples = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size)


class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # images go from 1x28x28 to 64x14x14
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)

        # images go from 64x14x14 to 128x7x7
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.dense1 = torch.nn.Linear(128 * 7 * 7, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x))).view(-1, 128 * 7 * 7)

        x = F.sigmoid(self.dense1(x))

        return x


class generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1 = torch.nn.Linear(128, 256)

        self.dense2 = torch.nn.Linear(256, 1024)

        self.dense3 = torch.nn.Linear(1024, 128 * 7 * 7)

        # from 128x7x7 to 64x14x14
        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # from 64x14x14 to 1x28x28
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

        self.bn1 = torch.BatchNorm1d(256)

        self.bn2 = torch.BatchNorm1d(1024)

        self.bn3 = torch.BatchNorm1d(128 * 7 * 7)

        self.bn4 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = F.relu(self.bn3(self.dense3(x))).view(-1, 128, 7, 7)

        x = F.relu(self.bn4(self.uconv1(x)))

        s = F.sigmoid(self.uconv(2))


# we use GPU ".cuda()"
d = discriminator().cuda()

g = generator().cuda()

# we define the hyper-parameters
no_epochs = 100

dlr = 0.0003

glr = 0.0003

# we use ADAM momentum
d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)

# we use ADAM momentum
g_optimizer = torch.optim.Adam(g.parameters(), lr=glr)

# training loop
for epoch in range(no_epochs):
    epochdcost = 0
    epochgcost = 0

    for k, (real_images,) in enumerate(train_samples):
        real_images = Variable(real_images).cuda()

        # Z is the noise input
        noise_input = Variable(torch.randn(batch_size, 128)).cuda()

        generated_images = g.forward(noise_input)

        # D(G(Z))
        gen_pred = d.forward(generated_images)

        real_pred = d.forward(real_images)

        # we use the binary cross-entropy cost
        dcost = -torch.sum(torch.log(real_pred) + torch.log(1 - gen_pred)) / batch_size

        gcost = -torch.sum(torch.log(gen_pred)) / batch_size

        d_optimizer.zero_grad()

        # we re-use the graph and we use "retain_graph"
        dcost.backward(retain_graph=True)

        d_optimizer.step()

        g_optimizer.zero_grad()

        gcost.backward()

        g_optimizer.step()

        epochdcost += dcost.data[0]

        epochgcost += gcost.data[0]

    epochcost /= 60000 / batch_size
    epochgcost /= 60000 / batch_size

    print('Epoch: ', epoch)

    print('Discriminator Cost: ', epochdcost)
    print('Generator Cost: ', epochgcost)

    dcosts.append(epochdcost)
    gcosts.append(epochgcost)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')

    ax.set_xlim(0, no_epochs)

    ax.plot(dcosts, 'b')

# file: https://drive.google.com/open?id=165pEQUXJUKoP6BZ8-MYyPo_UTRu2n7cB

# use: colaboratory
# https://drive.google.com/open?id=165pEQUXJUKoP6BZ8-MYyPo_UTRu2n7cB

