
import pyai
from pyai.backend import one_hot_encode
from pyai.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.datasets import mnist

def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255
    train_labels = one_hot_encode(train_labels, 10)
    test_images = test_images / 255
    test_labels = one_hot_encode(test_labels, 10)
    
    return (train_images, train_labels), (test_images, test_labels)

def train():
    (train_images, train_labels), validation_data = load_data()

    network = pyai.Network([
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.5),
        Dense(10, 'softmax')
    ])

    network.compile(optimiser='adam')

    network.fit(train_images, train_labels, 100, 15, validation_data=validation_data)

    network.save('mnist_network.pyai')

(train_images, train_labels), (test_images, test_labels) = load_data()

network = pyai.Network.load('mnist_network.pyai')

network.evaluate(test_images, test_labels, 100)