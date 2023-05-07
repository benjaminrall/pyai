import personallib.maths as m
from nn.layer import Layer
import time

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, layers = [], fileName = "", fromFile = False):
        self.fileName = fileName
        if fromFile:
            with open(fileName,"r") as f:
                data = [ line.strip().split(",") for line in f.readlines()]
                data[0:2] = [ [ int(n) for n in data[i] ] for i in range(2)]
                data[2:] = [ (n[0], float(n[1])) for n in data[2:] ]
            layers = data[0]
        self.inputLayer = Layer(layers[0])
        self.layers = [self.inputLayer]
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i], layers[i - 1]))
        self.outputLayer = self.layers[-1]
        self.outputCosts = [ 0 for i in range(len(self.outputLayer.values.matrix))]
        self.totalCost = 0
        self.gradientLength = 0
        for layer in self.layers:
            self.gradientLength += layer.weights.dimensions[0] * layer.weights.dimensions[1]
            self.gradientLength += layer.biases.dimensions[0] * layer.biases.dimensions[1]
        self.values = []
        self.layerValuePointers = []
        print(f"Network {layers} creation started")
        time1 = time.time()
        p = 0
        for layer in self.layers:
            self.layerValuePointers.append(p)
            for j, row in enumerate(layer.weights.matrix):
                for k, item in enumerate(row):
                    self.values.append((f'w:{j}:{k}', item))
                    p += 1
            for j, bias in enumerate(layer.biases.matrix):
                self.values.append((f'b:{j}', bias[0]))
                p += 1
        time2 = time.time()
        self.gradients = [0 for i in range(self.gradientLength)]
        print(f"Network created in {time2 - time1}")
        if fromFile:
            self.layerValuePointers = data[1]
            self.values = data[2:]
            self.map_values()
        self.averageCost = 1

    def set_values(self, layer, data, s = False):
        values = self.layers[layer].values
        for i in range(len(data)):
            if not s:
                values.set_row(i, [data[i]])
            else:
                values.set_row(i, [m.sigmoid(data[i])])

    def feedforward(self, input):
        self.set_values(0, input)
        for n in range(1, len(self.layers)):
            self.set_values(n, self.calculate_layer(n), True)

    def calculate_layer(self, n):
        layer = self.layers[n]
        a = [ row[0] for row in m.Matrix.add(m.Matrix.multiply(layer.weights, self.layers[n - 1].values), layer.biases).matrix ] 
        return a

    def get_output(self):
        print(self.outputLayer.values.display())

    def cost(self, i, e):
        return ((i - e) ** 2)

    def test(self, input, expected):
        self.feedforward(input)
        output = self.outputLayer.values.matrix
        for i in range(len(self.outputCosts)):
            self.outputCosts[i] = self.cost(output[i][0], expected[i])
        self.totalCost = sum(self.outputCosts)

    def train(self, inputs, expecteds, learningRate, progression = True):
        if len(inputs) != len(expecteds):
            return [0 for i in range(self.gradientLength)]
        n = len(inputs)
        self.gradients = [0 for i in range(self.gradientLength)]
        averageCost = 0
        for i in range(n):
            self.test(inputs[i], expecteds[i])
            averageCost += self.totalCost
            self.gradients = [a + b for a, b in zip(self.gradients, self.backwards_propagate(expecteds[i]))]
            if progression:
                print(f"Finished gradients {i + 1}/{n}", end = "\r")
        self.gradients = [i / n for i in self.gradients]
        self.values = [ (self.values[i][0], self.values[i][1] - ((learningRate/len(inputs)) * self.gradients[i])) for i in range(len(self.gradients))]
        self.map_values()
        averageCost = averageCost / n
        self.averageCost = averageCost
        if progression:
            print("")

    def map_values(self):
        p = 0
        for layer in self.layers:
            for j in range(len(layer.weights.matrix)):
                for i in range(len(layer.weights.matrix[j])):
                    layer.weights.matrix[j][i] = self.values[p][1]
                    p += 1
            for i in range(len(layer.biases.matrix)):
                layer.biases.matrix[i][0] = self.values[p][1]
                p += 1

    def weight_gradient(self, connectedNode, value, nodeGradient):
        return connectedNode * sigmoid_derivative(value) * nodeGradient

    def bias_gradinet(self, value, nodeGradient):
        return sigmoid_derivative(value) * nodeGradient

    def previous_node_gradient(self, weight, value, nodeGradient):
        return weight * sigmoid_derivative(value) * nodeGradient

    def backwards_propagate(self, expecteds):
        gradients = [0 for i in range(len(self.gradients))]
        outputIndex = self.layerValuePointers[-1]
        outputValues = self.values[outputIndex:]
        for layer in self.layers:
            layer.gradients = [ 0 for i in range(len(layer.gradients))]
        for i, value in enumerate(outputValues):
            v = value[0].split(':')
            j = int(v[1])
            if v[0] == "w":
                k = int(v[2])
                gradients[outputIndex + i] = self.weight_gradient(self.layers[-2].values.matrix[k][0], self.outputLayer.values.matrix[j][0], 2 * (self.outputLayer.values.matrix[j][0] - expecteds[j]))
                self.layers[-2].gradients[k] += self.previous_node_gradient(self.outputLayer.weights.matrix[j][k], self.outputLayer.values.matrix[j][0], 2 * (self.outputLayer.values.matrix[j][0] - expecteds[j]))
            else:
                gradients[outputIndex + i] = self.bias_gradinet(self.outputLayer.values.matrix[j][0], 2 * (self.outputLayer.values.matrix[j][0] - expecteds[j]))
        for L in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[L]
            previousLayer = self.layers[L - 1]
            layerIndex = self.layerValuePointers[L]
            layerValues = self.values[layerIndex:self.layerValuePointers[L + 1]]
            for i, value in enumerate(layerValues):
                v = value[0].split(':')
                j = int(v[1])
                if v[0] == "w":
                    k = int(v[2])
                    gradients[layerIndex + i] = self.weight_gradient(previousLayer.values.matrix[k][0], layer.values.matrix[j][0], layer.gradients[j])
                    previousLayer.gradients[k] += self.previous_node_gradient(layer.weights.matrix[j][k], layer.values.matrix[j][0], layer.gradients[j])
                else:
                    gradients[layerIndex + i] = self.bias_gradinet(layer.values.matrix[j][0], layer.gradients[j])
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

    def test_network(self, inputs, expecteds):
        correct = 0
        for x in range(len(inputs)):
            print(f"Testing: {x + 1}/{len(inputs)}", end = "\r")
            i = inputs[x]
            expected = expecteds[x]
            self.test(i, expected)
            output = [ value[0] for value in self.outputLayer.values.matrix]
            if expected[output.index(max(output))] == 1:
                correct += 1
        print(f"\nCorrect: {correct}/{len(inputs)}")