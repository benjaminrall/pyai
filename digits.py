import numpy as np
from layers import Dense
from network import Network
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images: np.ndarray = train_images / 255
test_images: np.ndarray = test_images / 255

train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)

def get_image(label: int) -> np.ndarray:
    result = np.zeros(10)
    result[label] = 1
    return result

test_labels = np.array([get_image(label) for label in test_labels])
train_labels = np.array([get_image(label) for label in train_labels])

network = Network([
    Dense(100, 'relu', 'glorot_uniform'),
    Dense(100, 'tanh'),
    Dense(10, 'sigmoid')
])

network.compile(
    (784,), "binary_cross_entropy"
)

network.fit(
    train_images, train_labels, 
    test_images, test_labels, 
    10, 0.02, 20
)
print(network.evaluate_accuracy(test_images, test_labels))