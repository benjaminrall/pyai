import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images: np.ndarray = train_images / 255
test_images: np.ndarray = test_images / 255

def get_image(label):
    result = np.zeros(10)
    result[label - 1] = 1
    return result

test_labels = np.array([get_image(label) for label in test_labels])
train_labels = np.array([get_image(label) for label in train_labels])

test_images = test_images.reshape(10000, 784)
test_labels = test_labels.reshape(10000, 10)

train_images = train_images.reshape(60000, 784)
train_labels = train_labels.reshape(60000, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),#from_logits=True),
    metrics=['accuracy'],
)

model.evaluate(test_images, test_labels, verbose=1)

model.summary()

model.fit(train_images, train_labels, epochs=20, batch_size=10)

model.evaluate(test_images, test_labels)