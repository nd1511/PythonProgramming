# PCA, Principal Component Analysis
# Dimensionality reduction using PCA

# We use matrices and transformations
# we can use 2 dimensions to visualize our transformation

# PCA and SVD
# dimensionality reduction

# we want to find the direction in which the variance is the largest
# we find the maximum-variance direction

# variance, sigma_squred = (1/M) \times \sum (x - \mu)^2
# where M data points

# covariance matrix, how does one feature vary as another feature varies
# we find the eigenvectors of the covariance matrix

# the diagonals of the covariance matrix equal to 1

# (1) compute the covariance matrix
# (2) eigenvalues and eigenvectors of the covariance matrix
# (3) keep the eigenvectors with the largest eigenvalues

# we now implement PCA

# use matplotlib
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# we use sklearn to load datasets
from sklearn import datasets



# we load the Iris dataset from sklearn
data = datasets.load_iris()

print(data)

# we now define the variables X and Y, our data
X = data.data
Y = data.target

m = X.shape[0]

def normalise(x):
    x_std = x - np.mean(x, axis=0)

    # we use np.std(.) to compute the standard deviation
    x_std = np.divide(x_std, np.std(x_std, axis=0))

    return x_std



def decompose(x):
    cov = np.matmul(x.T, x)

    print('\n Covariance matrix')
    print(cov)

    eig_vals, eig_vecs = np.linalg.eig(cov)

    print('\n Eigenvectors')
    print(eig_vecs)

    print('\n Eigenvalues')
    print(eig_vals)

    return eig_vals, eig_vecs, cov



# we now find which eigenvectors are important
def whicheigs(eig_vals):
    total = sum(eig_vals)

    # we use descending order, we use "sorted(eig_vals, reverse=True)"

    # we define the variance percentage
    var_percent = [(i/total)*100 for i in sorted(eig_vals, reverse=True)]

    cum_var_percent = np.cumsum(var_percent)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Variance along different principal components')

    ax.grid()

    plt.xlabel('Principal component')
    plt.ylabel('Percentage total variance accounted for')

    ax.plot(cum_var_percent, '-ro')

    ax.bar(range(len(eig_vals)), var_percent)

    plt.xticks(np.arange(len(eig_vals)), ('PC{}'.format(i) for i in range(len(eig_vals))))

    plt.show()



# we now call the functions
X_std = normalise(X)

eig_vals, eig_vecs, cov = decompose(X_std)

whicheigs(eig_vals)

def reduce(x, eig_vecs, dims):
    W = eig_vecs[:, :dims]

    print('\n Dimension reducing matrix')
    print(W)

    return np.matmul(x,W), W



colour_dict = {0:'r', 1:'g', 2:'b'}

colour_list = [colour_dict[i] for i in list(Y)]

def plotreduced(x):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # "x[:,0]" is the first principal component
    #ax.scatter(x[:,0], x[:,1], x[:,2])

    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colour_list)

    plt.grid()
    plt.show()

# we now call the functions
X_std = normalise(X)

eig_vals, eig_vecs, cov = decompose(X_std)

whicheigs(eig_vals)

X_reduced, transform = reduce(X_std, eig_vecs, 3)

# we plot the graph with the reduced data
plotreduced(X_reduced)



# Deep Generative Models
# GANs and VAEs, Generative Models

# random noise
# from random noise to a tensor

# We use batch normalisation.
# GANs are very difficult to train. Super-deep models. This is why we use batch normalisation.

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb



# Anomaly detection (AD)
# Unsupervised machine learning

# GANs for super-resolution
# Generative Adversarial Networks, GANs

# the BigGAN dataset
# BigGAN => massive dataset
# latent space, BigGAN, GANs

# down-sampling, sub-sample, pooling
# throw away samples, pooling, max-pooling

# partial derivatives
# loss function and partial derivatives

# https://github.com/Students-for-AI/The-Academy-of-AI
# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models

# Generator G and Discriminator D
# the loss function of the Generator G

# up-convolution
# We use a filter we do up-convolution with.

# use batch normalisation
# GANs are very difficult to train and this is why we use batch normalisation.

# We normalize across a batch.
# Mean across a batch. We use batches. Normalize across a batch.

# the ReLU activation function
# ReLU is the most common activation function. We use ReLU.

# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb



# use PyTorch
import torch

#import torch
import torchvision

from torchvision import datasets, transforms

# use matplotlib
import matplotlib.pyplot as plt

#import torch
#import torchvision

#from torchvision import transforms, datasets

import torch.nn.functional as F

#import matplotlib.pyplot as plt

#batch_size = 128

# download the training dataset
#train_data = datasets.FashionMNIST(root='fashiondata/',
#                                   transform=transforms.ToTensor(),
#                                   train=True,
#                                   download=True)

# we create the train data loader
#train_loader = torch.utils.data.DataLoader(train_data,
#                                           shuffle=True,
#                                           batch_size=batch_size)

batch_size = 100

train_data = datasets.FashionMNIST(root='fashiondata/',
                                 transform=transforms.ToTensor(),
                                 train=True,
                                 download=True
                                 )

train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True
                                           )

# combine the power of LSTM RNNs and GANs
# it is possible to use LSTM RNN together with GANs

# GANs and LSTM RNNs
# use LSTM RNNs together with GANs

# class for D and G
# we train the discriminator and the generator

# we make the discriminator
class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)  # 1x28x28-> 64x14x14
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 64x14x14-> 128x7x7
        self.dense1 = torch.nn.Linear(128 * 7 * 7, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))).view(-1, 128 * 7 * 7)
        x = F.sigmoid(self.dense1(x))
        return x

# this was for the discriminator
# we now do the same for the generator

# Generator G
class generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(256, 1024)
        self.dense3 = torch.nn.Linear(1024, 128 * 7 * 7)
        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 128x7x7 -> 64x14x14
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)  # 64x14x14 -> 1x28x28

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.bn3 = torch.nn.BatchNorm1d(128 * 7 * 7)
        self.bn4 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = F.relu(self.bn3(self.dense3(x))).view(-1, 128, 7, 7)
        x = F.relu(self.bn4(self.uconv1(x)))
        x = F.sigmoid(self.uconv2(x))
        return x

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb
# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# instantiate the model
d = discriminator()
g = generator()

# training hyperparameters
epochs = 100

# learning rate
dlr = 0.0003
glr = 0.0003

d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)
g_optimizer = torch.optim.Adam(g.parameters(), lr=glr)

dcosts = []
gcosts = []
plt.ion()
fig = plt.figure()
loss_ax = fig.add_subplot(121)
loss_ax.set_xlabel('Batch')
loss_ax.set_ylabel('Cost')
loss_ax.set_ylim(0, 0.2)
generated_img = fig.add_subplot(122)
plt.show()

# https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

# https://github.com/life-efficient/Academy-of-AI/tree/master/Lecture%2013%20-%20Generative%20Models
# use: https://github.com/life-efficient/Academy-of-AI/blob/master/Lecture%2013%20-%20Generative%20Models/GANs%20tutorial.ipynb

def train(epochs, glr, dlr):
    g_losses = []
    d_losses = []

    for epoch in range(epochs):

        # iteratre over mini-batches
        for batch_idx, (real_images, _) in enumerate(train_samples):

            z = torch.randn(batch_size, 128)  # generate random latent variable to generate images from
            generated_images = g.forward(z)  # generate images

            gen_pred = d.forward(generated_images)  # prediction of discriminator on generated batch
            real_pred = d.forward(real_images)  # prediction of discriminator on real batch

            dcost = -torch.sum(torch.log(real_pred)) - torch.sum(torch.log(1 - gen_pred))  # cost of discriminator
            gcost = -torch.sum(torch.log(gen_pred)) / batch_size  # cost of generator

            # train discriminator
            d_optimizer.zero_grad()
            dcost.backward(retain_graph=True)  # retain the computational graph so we can train generator after
            d_optimizer.step()

            # train generator
            g_optimizer.zero_grad()

            gcost.backward()
            g_optimizer.step()

            # give us an example of a generated image after every 10000 images produced
            if batch_idx * batch_size % 1000 == 0:
                g.eval()  # put in evaluation mode
                noise_input = torch.randn(1, 128)
                generated_image = g.forward(noise_input)

                generated_img.imshow(generated_image.detach().squeeze(), cmap='gray_r')
                g.train()  # put back into training mode

            dcost /= batch_size
            gcost /= batch_size

            print('Epoch: ', epoch, 'Batch idx:', batch_idx, '\tDisciminator cost: ', dcost.item(),
                  '\tGenerator cost: ', gcost.item())

            dcosts.append(dcost)
            gcosts.append(gcost)

            loss_ax.plot(dcosts, 'b')
            loss_ax.plot(gcosts, 'r')

            fig.canvas.draw()

#print(torch.__version__)
train(epochs, glr, dlr)

# We obtain:
# Epoch:  0 Batch idx: 0 	Disciminator cost:  1.3832124471664429 	Generator cost:  0.006555716972798109
# Epoch:  0 Batch idx: 1 	Disciminator cost:  1.0811840295791626 	Generator cost:  0.008780254982411861
# Epoch:  0 Batch idx: 2 	Disciminator cost:  0.8481155633926392 	Generator cost:  0.011281056329607964
# Epoch:  0 Batch idx: 3 	Disciminator cost:  0.6556042432785034 	Generator cost:  0.013879001140594482
# Epoch:  0 Batch idx: 4 	Disciminator cost:  0.5069876909255981 	Generator cost:  0.016225570812821388
# Epoch:  0 Batch idx: 5 	Disciminator cost:  0.4130948781967163 	Generator cost:  0.018286770209670067
# Epoch:  0 Batch idx: 6 	Disciminator cost:  0.33445805311203003 	Generator cost:  0.02015063539147377
# Epoch:  0 Batch idx: 7 	Disciminator cost:  0.279323011636734 	Generator cost:  0.021849267184734344
# Epoch:  0 Batch idx: 8 	Disciminator cost:  0.2245958000421524 	Generator cost:  0.02352861315011978
# Epoch:  0 Batch idx: 9 	Disciminator cost:  0.18664218485355377 	Generator cost:  0.025215130299329758
# Epoch:  0 Batch idx: 10 	Disciminator cost:  0.14700829982757568 	Generator cost:  0.02692217379808426

# generate random latent variable to generate images
z = torch.randn(batch_size, 128)

# generate images
im = g.forward(z)

plt.imshow(im)

