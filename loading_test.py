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

network = pyai.Network.load('dense_network.pyai')
score = network.evaluate(test_images, test_labels, 100)