
# we use Keras and TensorFlow
# we use: https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/

# we use the book of F. Chollet: Deep Learning with Python
# we use: http://amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438/ref=sr_1_1?ie=UTF8&qid=1523486008&sr=8-1&keywords=chollet

import keras

from keras.datasets import mnist



# we use the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(' ')

print(test_images.shape)
print(len(test_labels))
print(test_labels)



from keras import models

from keras import layers



network = models.Sequential()

# we add the dense fully-connected layers
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

# we add the output layer, the output softmax layer
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255



from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)



network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)





