from keras.datasets import mnist
from pyai.nn.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from pyai.backend import one_hot_encode
import pyai
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255
train_labels = one_hot_encode(train_labels, 10)
test_images = test_images / 255
test_labels = one_hot_encode(test_labels, 10)

np.random.seed(0)

network = pyai.nn.Network([
    Conv2D(8, (2, 2), activation='relu'),
    Conv2D(8, (4, 4), (3, 3), 'relu'),
    Flatten(),
    Dense(10, 'relu'),
    Dense(10, 'softmax')
])
network.build((28,28,1))
network.summary()

network.compile(optimiser='adam')

network.fit(train_images, train_labels, 100, 10, validation_data=(test_images, test_labels))

#network.save('dense_network.pyai')
#network = pyai.Network.load('dense_network.pyai')

network.evaluate(test_images, test_labels, 100)