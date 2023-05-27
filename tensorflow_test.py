from keras.datasets import mnist
from keras.layers import Dense
from keras.backend import one_hot
from tensorflow import keras
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = (train_images / 255).reshape(-1, 784)
train_labels = one_hot(train_labels, 10)
test_images = (test_images / 255).reshape(-1, 784)
test_labels = one_hot(test_labels, 10)

np.random.seed(0)
network = keras.Sequential([
    Dense(100, 'relu'),
    Dense(100, 'sigmoid'),
    Dense(10, 'softmax'),
])

network.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(learning_rate=0.1),
    metrics=['accuracy']
)

network.fit(
    train_images, train_labels,
    10, 20
)

network.evaluate(test_images, test_labels)