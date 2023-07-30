from keras.datasets import mnist
from pyai.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from pyai.backend import one_hot_encode
import pyai
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255
train_labels = one_hot_encode(train_labels, 10)
test_images = test_images / 255
test_labels = one_hot_encode(test_labels, 10)

np.random.seed(0)

network = pyai.Network([
    Conv2D(32, (3, 3), 'relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), 'relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.5),
    Dense(10, 'softmax')
])

network.build((28, 28, 1))
network.summary()

network.compile(
    loss='categorical_crossentropy',
    optimiser='nadam'
)


network.fit(train_images, train_labels, 100, 15, test_images, test_labels)