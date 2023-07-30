from keras.datasets import mnist
from keras.layers import Dense, AveragePooling2D, MaxPooling2D, Conv2D, Flatten, Dropout
import keras.activations
import keras.backend
import keras.losses
import keras.optimizers
import keras.regularizers
from keras.backend import one_hot
from tensorflow import keras
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = (train_images / 255).reshape(-1, 784)
train_labels = one_hot(train_labels, 10)
test_images = (test_images / 255).reshape(-1, 784)
test_labels = one_hot(test_labels, 10)
network = keras.Sequential([
    Dense(100, 'relu'),
    Dense(100, 'sigmoid'),
    Dense(10, 'softmax'),
])

np.random.seed(0)

network.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

network.build(train_images.shape)

network.summary()

network.fit(train_images, train_labels, 10, 1)

network.evaluate(test_images, test_labels, 1)