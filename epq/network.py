import random       # Random module for shuffling an array
import numpy as np  # Numpy module for fast linear algebra
import pickle       # Pickle module for serialising the neural network data into binary files

from progressbar import ProgressBar # Progress bar class for informative console progress bars

# Applies the sigmoid function to every element in an array
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Applies the sigmoid derivative to every element in an array
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, layers, filePath = ""):
        self.layers = layers    # Array containing the sizes of each layer in the network
        self.size = len(layers) # The amount of layers in the network

        self.biases = [ np.random.randn(n, 1) for n in layers[1:] ] # An array containing 2 dimensional numpy arrays for each layer's biases
        self.weights = [np.random.randn(layers[i], layers[i - 1]) for i in range(1, len(layers)) ]  # An array containing 2 dimensional numpy arrays for each layer's weights

        self.path = filePath    # File path for saving the neural network
    
    # Goes through each layer and uses matrix multiplication with the weights and biases in 
    # order to calculate the output of the neural network for a given set of inputs
    def feedforward(self, inputs):
        for w, b in zip(self.weights, self.biases):
            inputs = sigmoid(b + np.dot(w, inputs))
        return inputs

    # Trains the neural network using stochastic gradient descent over a specified number of epochs to improve its accuracy
    def train(self, trainingData, epochs, miniBatchSize, eta, testingData = None, saving = False):
        for i in range(epochs):
            print("{:-^90s}".format(f"  Epoch {i + 1}  "))
            progress = ProgressBar("Training", len(trainingData))
            random.shuffle(trainingData)
            miniBatches = [ trainingData[b:b + miniBatchSize] for b in range(0, len(trainingData), miniBatchSize) ]
            for batchNumber, miniBatch in enumerate(miniBatches):
                nablaBiases = [ np.zeros(b.shape) for b in self.biases ]
                nablaWeights = [ np.zeros(w.shape) for w in self.weights ]
                for dataNumber, data in enumerate(miniBatch):
                    deltaNablaWeights, deltaNablaBiases = self.backprop(data[0], data[1])
                    for b in range(len(nablaBiases)):
                        nablaBiases[b] = nablaBiases[b] + deltaNablaBiases[b]
                    for w in range(len(nablaWeights)):
                        nablaWeights[w] = nablaWeights[w] + deltaNablaWeights[w]
                    progress.update(batchNumber * miniBatchSize + dataNumber + 1)
                for b in range(len(self.biases)):
                    self.biases[b] = self.biases[b] - (eta / miniBatchSize) * nablaBiases[b]
                for w in range(len(self.weights)):
                    self.weights[w] = self.weights[w] - (eta / miniBatchSize) * nablaWeights[w]
            if testingData:
                correct, cost = self.test(testingData)
                print(f'  -  Testing accuracy: {round((correct / len(testingData)) * 100, 1)}% {correct} / {len(testingData)}  | Average cost: {round(cost[0], 8)}')
            else:
                print(f'  -  Epoch {i + 1} complete.')
            if self.path != "" and saving:
                self.save()

    # Evaluates the performance of a network by using a set of training data and 
    # returning the amount of correct classification and the average cost of the network
    def test(self, testingData):
        correct = 0
        progress = ProgressBar("Testing", len(testingData))
        costs = []
        for i, data in enumerate(testingData):
            output = self.feedforward(data[0])
            if np.argmax(output) == np.argmax(data[1]):
                correct += 1
            costs.append(self.calculate_cost(output, data[1]))
            progress.update(i + 1)
        return correct, sum(costs) / len(costs)

    # Calculates the cost of a set of outputs using the squared difference
    def calculate_cost(self, output, expected):
        return sum([(output[i] - expected[i]) ** 2 for i in range(len(output))])

    # Performs a singular pass of the backpropagation algorithm for a set of inputs
    def backprop(self, input, expected):
        # Forwards pass through the network to store all of the activation and 'z' values
        # for each layer in the network
        activation = input
        activations = [input]
        zVectors = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zVectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backwards pass through the network to calculate the gradient for the weights
        # and biases in order to minimise the cost function
        nablaBiases = [np.zeros(b.shape) for b in self.biases]
        nablaWeights = [np.zeros(w.shape) for w in self.weights]
        delta = self.cost_derivative(activations[-1], expected) * sigmoid_derivative(zVectors[-1])
        nablaBiases[-1] = delta
        nablaWeights[-1] = np.dot(delta, np.transpose(activations[-2]))
        for layer in range(2, self.size):
            z = zVectors[-layer]
            sd = sigmoid_derivative(z)
            delta = np.dot(np.transpose(self.weights[-layer + 1]), delta) * sd
            nablaBiases[-layer] = delta
            nablaWeights[-layer] = np.dot(delta, np.transpose(activations[-layer - 1]))
        return (nablaWeights, nablaBiases)

    # Finds the derivative of the cost function for a singular output and its expected value
    def cost_derivative(self, output, expected):
        return 2 * (output - expected)

    # Serialises the weights and biases and stores them in a file
    def save(self, fileName = ""):
        if fileName == "" and self.path == "":
            print("No file name specified or stored for the network")
            return
        elif fileName == "":
            fileName = self.path
        with open(fileName + ".weights", "wb") as fw:
            pickle.dump(self.weights, fw)
        with open(fileName + ".biases", "wb") as fb:
            pickle.dump(self.biases, fb)

    # Loads weights and biases from a specified file and returns a new neural network object
    @staticmethod
    def load(fileName):
        with open(fileName + ".weights", "rb") as fw:
            weights = pickle.load(fw)
        with open(fileName + ".biases", "rb") as fb:
            biases = pickle.load(fb)
        layers = [weight.shape[1] for weight in weights] + [biases[-1].shape[0]]
        network = NeuralNetwork(layers, fileName)
        network.weights = weights
        network.biases = biases
        return network