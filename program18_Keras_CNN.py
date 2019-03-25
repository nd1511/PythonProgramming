# we use CNNs, we use Keras

# import Keras layers
from keras import layers

# import Keras models
from keras import models

# we use Sequential
model = models.Sequential()

# we use: "input_shape = (28, 28, 1)"

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#print(model.summary())

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())



# we use the MNIST dataset

# import MNIST
from keras.datasets import mnist

# import categorical
from keras.utils import to_categorical

# define the training data and the test data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#from keras.datasets import mnist

# we use tuples, (..., ...)
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#print('')
#print(train_images.ndim)
#print(train_images.shape)
#print(train_images.dtype)

#digit = train_images[4]

#import matplotlib.pyplot as plt

#plt.imshow(digit, cmap=plt.cm.binary)

#plt.show()
#plt.close()

#my_slice = train_images[10:100]

#print('')
#print(my_slice.shape)



train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)

# we find the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

# accuracy
print(test_acc)



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



# we use PyTorch
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

# define the batch size
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
#epochs = 100

#epochs = 100
epochs = 10

# learning rate
#dlr = 0.0003
#glr = 0.0003

dlr = 0.003
glr = 0.003

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
            #if batch_idx * batch_size % 10000 == 0:

            # give us an example of a generated image after every 20 images produced
            if batch_idx % 20 == 0:
                g.eval()  # put in evaluation mode
                noise_input = torch.randn(1, 128)
                generated_image = g.forward(noise_input)

                generated_img.imshow(generated_image.detach().squeeze(), cmap='gray_r')

                # pause for some seconds
                plt.pause(5)

                # put back into training mode
                g.train()

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

# Epoch:  0 Batch idx: 32 	Disciminator cost:  0.2818330228328705 	Generator cost:  0.022729918360710144
# Epoch:  0 Batch idx: 33 	Disciminator cost:  0.7310256361961365 	Generator cost:  0.05649786815047264
# Epoch:  0 Batch idx: 34 	Disciminator cost:  0.31759023666381836 	Generator cost:  0.02075548656284809
# Epoch:  0 Batch idx: 35 	Disciminator cost:  0.35554683208465576 	Generator cost:  0.018939709290862083
# Epoch:  0 Batch idx: 36 	Disciminator cost:  0.07700302451848984 	Generator cost:  0.04144695773720741
# Epoch:  0 Batch idx: 37 	Disciminator cost:  0.08900360018014908 	Generator cost:  0.05888563022017479
# Epoch:  0 Batch idx: 38 	Disciminator cost:  0.0921328067779541 	Generator cost:  0.0593753345310688
# Epoch:  0 Batch idx: 39 	Disciminator cost:  0.09943853318691254 	Generator cost:  0.05279992148280144
# Epoch:  0 Batch idx: 40 	Disciminator cost:  0.2455407679080963 	Generator cost:  0.036564696580171585
# Epoch:  0 Batch idx: 41 	Disciminator cost:  0.10074597597122192 	Generator cost:  0.03721988573670387
# Epoch:  0 Batch idx: 42 	Disciminator cost:  0.07906078547239304 	Generator cost:  0.04363853484392166

# Epoch:  0 Batch idx: 444 	Disciminator cost:  0.06787727028131485 	Generator cost:  0.04046594724059105
# Epoch:  0 Batch idx: 445 	Disciminator cost:  0.07139576226472855 	Generator cost:  0.03837932273745537
# Epoch:  0 Batch idx: 446 	Disciminator cost:  0.08202749490737915 	Generator cost:  0.039551254361867905
# Epoch:  0 Batch idx: 447 	Disciminator cost:  0.12328958511352539 	Generator cost:  0.03817861154675484
# Epoch:  0 Batch idx: 448 	Disciminator cost:  0.06865841150283813 	Generator cost:  0.03938257694244385

# generate random latent variable to generate images
z = torch.randn(batch_size, 128)

# generate images
im = g.forward(z)
# use "forward(.)"

plt.imshow(im)

