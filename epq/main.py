import random
from network import NeuralNetwork
import numpy as np
from matplotlib import pyplot

# HANDWRITTEN DIGITS
def handwritten_digits():
    import mnist_loader

    # Loads data for training and testing the neural network on the MNIST database
    trainingData = mnist_loader.load_training_data()
    testingData = mnist_loader.load_testing_data()

    # Creates a neural network and trains it to recognise handwritten digits over 30 epochs
    network = NeuralNetwork([784, 16, 16, 10], "SavedNetworks/handwritten_digits")
    network.train(trainingData, 30, 10, 0.1, testingData)

    # Saves the weights and biases to files
    network.save()

# CATS VS DOGS
def cats_vs_dogs():
    import catsanddogs_loader

    # Methods for processing and formatting image data to be used in the neural network
    # catsanddogs_loader.process_data("CatsAndDogsData/training")
    # catsanddogs_loader.format_photos("CatsAndDogsData/training", 25000)

    # Loads data for training the neural network on a set of 25000 cat and dog images
    trainingData = catsanddogs_loader.load_data()

    # Creates a neural network and trains it to recognise cats and dogs over 30 epochs
    network = NeuralNetwork([10000, 200, 200, 2], "SavedNetworks/cats_vs_dogs")
    network.train(trainingData[:10000] + trainingData[12500:22500], 30, 10, 0.1, trainingData[10000:12500] + trainingData[22500:])

    # Saves the weights and biases to files
    network.save()
    
# MUSIC GENRES
def music_genres():
    import gtzan_loader

    # Method for processing and formatting music clip data to be used in the neural network
    # gtzan_loader.process_data("MusicGenresData/training")

    # Loads data for training the neural on a set of 1000 music clips
    trainingData = gtzan_loader.load_data()
    random.shuffle(trainingData)

    # Creates a neural network and trains it to recognise music genres over 30 epochs
    network = NeuralNetwork([40000, 1000, 400, 10], "SavedNetworks/music_genres")
    network.train(trainingData[:900], 30, 10, 0.1, trainingData[900:], True)

    # Saves the weights and biases to files
    network.save()

handwritten_digits()
