import numpy as np
from keras.datasets import mnist
import tensorflow as tf
import keras.backend as K
from keras.datasets import *
from tensorflow.python import training
from tensorflow.python.framework.ops import Tensor

HANDWRITING_PATH = "handwriting_weights"

# MNIST HANDWRITING
def train_handwriting(model):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255
    test_images = test_images / 255

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=1)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest Accuracy: {test_acc}")

    model.save_weights(HANDWRITING_PATH)

def test_handwriting(model):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255
    test_images = test_images / 255

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest Accuracy: {test_acc}")

handwriting_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

train_handwriting(handwriting_model)
test_handwriting(handwriting_model)