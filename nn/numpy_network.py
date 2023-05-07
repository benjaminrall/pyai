import numpy as np
from numpy.lib.function_base import average
from numpy_layer import Layer
import math
import time

def sigmoid(x): # Calculates the sigmoid function of x
    return 1 / (1 + (math.e ** -x))

def sigmoid_derivative(sx): # Calculates the derivative of the sigmoid function of x, assuming sigmoid has already been applied to x
    return sx * (1 - sx)

class NeuralNetwork():
    def __init__(self, layers = [], fileName = "", fromFile = False):
        self.fileName = fileName
        if fromFile:    # Loads data from file, if required
            with open(fileName, "r") as f:
                data = [ line.strip().split(",") for line in f.readlines()]
                data[0:2] = [ [ int(n) for n in data[i] ] for i in range(2)]
                data[2:] = [ (n[0], float(n[1])) for n in data[2:] ]
            layers = data[0]
        print(f"Network {layers} creation started") 
        startTime = time.time() # Keeps track of the time creation started
        self.inputLayer = Layer(layers[0])  # Creates input layer
        self.layers = [self.inputLayer]
        for i in range(1, len(layers)):     # Creates all layers
            self.layers.append(Layer(layers[i], layers[i - 1]))
        self.outputLayer = self.layers[-1]  # Sets output layer
        self.outputCosts = [ 0 for i in range(len(self.outputLayer.values)) ] # Costs of the output layer
        self.totalCost = 0  # Total cost of the previous input
        self.averageCost = 100  # Average cost of all inputs over a training iteration
        self.gradientLength = 0 # Holds the length of the gradient function
        for layer in self.layers:   # Finds the gradient function length
            self.gradientLength += layer.weights.shape[0] * layer.weights.shape[1]
            self.gradientLength += layer.biases.shape[0] * layer.biases.shape[1]
        self.gradients = [ 0 for i in range(self.gradientLength) ]  # holds individual gradients to use in backwards propagation
        self.values = []    # every single weight/bias value in the network
        self.layerValuePointers = []    # list of pointers to the beginning of each layers values in values
        p = 0   # pointer to keep find the layer pointers
        for layer in self.layers:   # adds every value to values, and finds pointers
            self.layerValuePointers.append(p)
            for j, row in enumerate(layer.weights):
                for k, item in enumerate(row):
                    self.values.append((f'w:{j}:{k}', item))
                    p += 1
            for j, bias in enumerate(layer.biases):
                self.values.append((f'b:{j}', bias[0]))
                p += 1
        if fromFile:    # remaps values and pointers if reading from a file
            self.layerValuePointers = data[1]
            self.values = data[2:]
            self.map_values()
        print(f"Network created in {round(time.time() - startTime, 1)} seconds")    # Information about network creation
    
    def set_values(self, layerIndex, data, useSigmoid = False):    # Sets the values for some layer to a list of given values
        values = self.layers[layerIndex].values
        for i in range(len(data)):
            if useSigmoid:
                values[i] = [sigmoid(data[i])]
            else:
                values[i] = [data[i]]

    def map_values(self):   # Maps values from the values list to the weights and biases in the network
        p = 0
        for layer in self.layers:
            for j in range(len(layer.weights)):
                for k in range(len(layer.weights[j])):
                    layer.weights[j][k] = self.values[p][1]
                    p += 1
            for j in range(len(layer.biases)):
                layer.biases[j][0] = self.values[p][1]
                p += 1

    def calculate_layer(self, layerIndex):   # Calculates the activation values of a layer, without sigmoid
        layer = self.layers[layerIndex]
        previousLayer = self.layers[layerIndex - 1]
        activations = [ row[0] for row in np.dot(layer.weights, previousLayer.values) + layer.biases ]
        return activations

    def feed_forward(self, inputValue):  # Feeds an input forward through the network
        self.set_values(0, inputValue)
        for i in range(1, len(self.layers)):
            self.set_values(i, self.calculate_layer(i), True)

    def cost(self, outputValue, expectedValue): # Uses the sum-of-squares cost function to find the cost for a given output
        return (outputValue - expectedValue) ** 2

    def test(self, inputValue, expectedValue):    # Feeds an input through the network, and calculates its cost
        self.feed_forward(inputValue)
        output = self.outputLayer.values
        for i in range(len(output)):
            self.outputCosts[i] = self.cost(output[i][0], expectedValue[i])
        self.totalCost = sum(self.outputCosts)

    def test_network(self, inputValues, expectedValues, displayProgression = True):    # Tests inputs and expected values, and finds out how many are correct
        correct = 0
        averageCost = 0
        print("Testing: ", end = "\r")
        for i in range(len(inputValues)):
            inputValue = inputValues[i]
            expectedValue = expectedValues[i]
            self.test(inputValue, expectedValue)
            averageCost += self.totalCost
            output = [ value[0] for value in self.outputLayer.values ]
            if displayProgression:
                print("{:^14s}{:^14s}".format("Testing", f"{i + 1}/{len(inputValues)}"), end = "\r")
                # print(f"Testing: {i + 1}/{len(inputValues)}", end = "\r")
            if expectedValue[np.argmax(output)] == 1:
                correct += 1
        averageCost /= len(inputValues)
        print("{:^14s}{:^14s}{:^14s}{:^14s}{:^14s}".format("Testing", f"{i + 1}/{len(inputValues)}", f"{correct}/{len(inputValues)}", f"{round((correct / len(inputValues)) * 100, 2)}%", f"{round(averageCost, 10)}"), end="", flush=True)

    def train(self, inputValues, expectedValues, learningRate, displayProgression = True):  # Uses backward propagation to train a network for a given set of inputs and outputs, for one epoch
        n = len(inputValues)
        self.gradients = np.zeros(self.gradientLength)
        averageCost = 0
        correct = 0
        print("Training: ", end = "\r")
        for i in range(n):
            self.test(inputValues[i], expectedValues[i])
            averageCost += self.totalCost
            self.gradients = self.gradients + self.backwards_propagate(expectedValues[i])
            if displayProgression:
                print("{:^14s}{:^14s}".format("Training", f"{i + 1}/{n}"), end = "\r")
                #print(f"Training: {i + 1}/{n}", end = "\r")
            if expectedValues[i][np.argmax([value[0] for value in self.outputLayer.values])]:
                correct += 1
        self.gradients = (self.gradients / n) * learningRate
        self.values = [ (self.values[i][0], self.values[i][1] - self.gradients[i]) for i in range(len(self.gradients)) ]
        self.map_values()
        self.averageCost = averageCost / n  # Average cost over the epoch
        print("{:^14s}{:^14s}{:^14s}{:^14s}".format("Training", f"{i + 1}/{n}", f"{correct}/{len(inputValues)}", f"{round((correct / len(inputValues)) * 100, 2)}%"), end="", flush=True)
    
    def weight_gradient(self, connectedNode, value, nodeGradient):
        return connectedNode * sigmoid_derivative(value) * nodeGradient

    def bias_gradient(self, value, nodeGradient):
        return sigmoid_derivative(value) * nodeGradient

    def previous_node_gradient(self, weight, value, nodeGradient):
        return weight * sigmoid_derivative(value) * nodeGradient

    def backwards_propagate(self, expectedValues):
        gradients = np.zeros(self.gradientLength)
        for layer in self.layers[1:-1]:
            layer.gradients = [ 0 for i in range(len(layer.gradients)) ]
        self.outputLayer.gradients = [ 2 * (self.outputLayer.values[j][0] - expectedValues[j]) for j in range(len(self.outputLayer.gradients)) ]
        for L in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[L]
            previousLayer = self.layers[L - 1]
            layerIndex = self.layerValuePointers[L]
            layerValues = self.values[layerIndex:]
            if L < len(self.layers) - 1:
                layerValues = layerValues[:self.layerValuePointers[L + 1]]
            for i, value in enumerate(layerValues):
                v = value[0].split(":")
                j = int(v[1])
                if v[0] == "w":
                    k = int(v[2])
                    gradients[layerIndex + i] = self.weight_gradient(previousLayer.values[k][0], layer.values[j][0], layer.gradients[j])
                    previousLayer.gradients[k] += self.previous_node_gradient(layer.weights[j][k], layer.values[j][0], layer.gradients[j])
                else:
                    gradients[layerIndex + i] = self.bias_gradient(layer.values[j][0], layer.gradients[j])
        return gradients

    def save(self, fileName = ""):
        if self.fileName == "" and fileName == "":
            print("Failed to save - no file name provided")
            return
        elif fileName != "":
            self.fileName = fileName
        data = []
        layers = ""
        for layer in self.layers:
            layers += str(layer.size) + ","
        layers = layers[:-1]
        data.append(layers)
        pointers = ""
        for pointer in self.layerValuePointers:
            pointers += str(pointer) + ","
        pointers = pointers[:-1]
        data.append(pointers)
        for value in self.values:
            v = ""
            v += value[0] + ","
            v += str(value[1])
            data.append(v)
        with open(self.fileName, "w") as f:
            f.writelines([line + "\n" for line in data])