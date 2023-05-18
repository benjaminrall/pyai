from keras.datasets import mnist
from pyai.layers import Dense
from pyai.backend import one_hot_encode
import pyai

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = (train_images / 255).reshape(-1, 784)
train_labels = one_hot_encode(train_labels, 10)
test_images = (test_images / 255).reshape(-1, 784)
test_labels = one_hot_encode(test_labels, 10)

network = pyai.Network([
    Dense(100, 'relu'),
    Dense(100, 'sigmoid'),
    Dense(10, 'softmax'),
])

network.compile(
    input_shape=(784,),
    loss='categorical_crossentropy'
)

network.fit(
    train_images, train_labels,
    test_images, test_labels,
    10, 0.01, 20
)
