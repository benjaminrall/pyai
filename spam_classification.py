import numpy as np
from network import Network
from layers import *

np.random.seed(0)

total = 1500
a0 = 0 
a1 = 0

training_spam = np.loadtxt(open("spam/training_spam.csv"), delimiter=",").astype(int)
testing_spam = np.loadtxt(open("spam/testing_spam.csv"), delimiter=",").astype(int)

train_data = training_spam[:, 1:]
train_labels = training_spam[:, 0]
test_data = testing_spam[:, 1:]
test_labels = testing_spam[:, 0]

train_labels = train_labels.reshape(train_labels.size, 1)
test_labels = test_labels.reshape(test_labels.size, 1)

data = np.append(train_data, test_data).reshape(total, 54)
labels = np.append(train_labels, test_labels).reshape(total, 1)


def convert_label(label):
    label_array = np.zeros(2)
    label_array[label] = 1
    return label_array

labels = np.array([convert_label(label) for label in labels])

for run in range(100):
    p = np.random.permutation(total)
    data, labels = data[p], labels[p]

    split = 1250

    train_data = data[:split]
    train_labels = labels[:split]
    test_data = data[split:]
    test_labels = labels[split:]

    network = Network([
        Dense(54, 'sigmoid'),
        Dense(10, 'sigmoid'),
        Dense(5, 'sigmoid'),
        Dense(2, 'sigmoid')
    ])

    network.compile(
        train_data.shape[1:], 'binary_cross_entropy'
    )

    network.fit(train_data, train_labels, test_data, test_labels, 10, 0.005, 100, False)

    a0 += network.evaluate_accuracy(train_data, train_labels)
    a1 += network.evaluate_accuracy(test_data, test_labels)
    print(a0, a1)

print(f"Average a0: {a0}, Average a1: {a1}")