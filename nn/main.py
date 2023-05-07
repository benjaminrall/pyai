from numpy_network import NeuralNetwork
import pygame
import os
from personallib.camera import Camera
import pickle
from matplotlib import pyplot
import numpy as np
import time

# Constants
WIN_WIDTH = 900
WIN_HEIGHT = 650
FRAMERATE = 480
ICON_IMG = pygame.image.load(os.path.join("imgs", "icon.png"))

# Pygame Setup
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Digit Recognition")
pygame.display.set_icon(ICON_IMG)
clock = pygame.time.Clock()
drawingSize = 18
drawingSurface = pygame.Surface((28 * drawingSize, 28 * drawingSize))
pygame.font.init()
font = pygame.font.SysFont("georgia", 30)
resultText = font.render("Result: ", True, (0, 0, 0))
certaintyText = font.render("Certainty: ", True, (0, 0, 0))

# Objects
network = NeuralNetwork([784, 16, 10], "saved_networks/digits4.network", True)

# Training
def train(inputs, outputs, display = False):

    test_inputs, test_outputs = setup_testing(10000)

    epochs = int(input("How many epochs would you like to run for: "))

    print("{:-^84s}\n{:^84s}\n{:^14s}{:^14s}{:^14s}{:^14s}{:^14s}{:^14s}\n{:-^84s}".format("", f"Epoch {0}", "Type", "Progress", "Correct", "Accuracy", "Cost", "Time", ""))
    startTime = time.time()
    network.test_network(test_inputs, test_outputs, display)
    print("{:>14s}".format(f"{int(round(time.time() - startTime, 1))} seconds"))

    for i in range(epochs):
        print("{:-^84s}\n{:^84s}\n{:^14s}{:^14s}{:^14s}{:^14s}{:^14s}{:^14s}\n{:-^84s}".format("", f"Epoch {i + 1}", "Type", "Progress", "Correct", "Accuracy", "Cost", "Time", ""))
        startTime = time.time()
        network.train(inputs, outputs, 5, display)
        print("{:^14s}{:>14s}".format(f"{round(network.averageCost, 10)}", f"{int(round(time.time() - startTime, 1))} seconds"))
        network.save()
        startTime = time.time()
        network.test_network(test_inputs, test_outputs, display)
        print("{:>14s}".format(f"{int(round(time.time() - startTime, 1))} seconds"))

def setup_testing(n):
    from tensorflow.keras.datasets import mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    test_inputs = []
    test_outputs = []
    for i in range(n):
        x_input = [row[j] for row in test_x[i] for j in range(28)]
        y_output = [0 for i in range(10)]
        y_output[test_y[i]] = 1
        test_inputs.append(x_input)
        test_outputs.append(y_output)
    for i in range(len(test_inputs)):
        for j in range(len(test_inputs[i])):
            test_inputs[i][j] /= 255
    return test_inputs, test_outputs

test_inputs, test_outputs = setup_testing(100)
startTime = time.time()
network.test_network(test_inputs, test_outputs, True)
print("{:>14s}".format(f"{int(round(time.time() - startTime, 1))} seconds"))

def draw_plot(i):
    pyplot.subplot(335)
    pyplot.imshow(train_x[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()

# Variables
running = True
try:
    with open("handwriting/inputs.data", "rb") as f:
        inputs = pickle.load(f)
    with open("handwriting/outputs.data", "rb") as f:
        outputs = pickle.load(f)
except:
    print("failed to load from file, loading from dataset")
    from tensorflow.keras.datasets import mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    n = int(input("How much testing data to load: "))
    inputs = []
    for input in train_x[:n]:
        inputs.append([ input[i][j] / 255 for i in range(28) for j in range(28) ])
#train(inputs[:1875], outputs[:1875], True)
#exit()
canvas = [[0 for i in range(28)] for j in range(28)]
buttons = {1: False, 3: False}

def draw_canvas(canvas, surface, size):
    for r, row in enumerate(canvas):
        for c, col in enumerate(row):
            col = (-col) + 1
            pygame.draw.rect(surface, (255 * col, 255 * col, 255 * col), (c * size, r * size, size, size))

def in_canvas(pos, surface, size):
    xmin = (WIN_HEIGHT // 2) - (surface.get_height() // 2)
    ymin = (WIN_HEIGHT // 2) - (surface.get_height() // 2)
    xmax = xmin + (size * 28)
    ymax = ymin + (size * 28)
    return pos[0] >= xmin and pos[0] < xmax and pos[1] >= ymin and pos[1] < ymax

def get_canvas_index(pos, surface, size):
    xmin = (WIN_HEIGHT // 2) - (surface.get_height() // 2)
    ymin = (WIN_HEIGHT // 2) - (surface.get_height() // 2)
    pos = ((pos[0] - xmin) // size, (pos[1] - ymin) // size)
    return pos[1], pos[0]

def draw(canvas, indexes):
    for i in range(-1, 2):
        for j in range(-1, 2):
            row = indexes[0] + i
            col = indexes[1] + j
            if 0 <= row < 28 and 0 <= col < 28 and canvas[row][col] != 1:
                canvas[row][col] += (1 - canvas[row][col]) / 4
    canvas[indexes[0]][indexes[1]] = 1

def erase(canvas, indexes):
    for i in range(-1, 2):
        for j in range(-1, 2):
            row = indexes[0] + i
            col = indexes[1] + j
            if 0 <= row < 28 and 0 <= col < 28:
                canvas[row][col] -= canvas[row][col] / 2
    canvas[indexes[0]][indexes[1]] = 0

def predict(canvas):
    network.feed_forward([row[j] for row in canvas for j in range(28)])
    values = network.outputLayer.values
    answer = np.argmax(values)
    resultText = font.render(f"Result: {answer}", True, (0, 0, 0))
    certaintyText = font.render(f"Certainty: {round(values[answer][0] * 100, 1)}%", True, (0, 0, 0))
    return resultText, certaintyText

# Main Loop
if __name__ == '__main__':
    win.fill((150, 150, 150))
    dataIndex = 0
    while running:

        dt = clock.tick(FRAMERATE) * 0.001

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in buttons:
                    buttons[event.button] = True
                elif event.button == 2:
                    canvas = [[0 for i in range(28)] for j in range(28)]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in buttons:
                    buttons[event.button] = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    resultText, certaintyText = predict(canvas)
                elif event.key == pygame.K_n:
                    nextInput = inputs[dataIndex]
                    canvas = [[nextInput[(28 * i) + j] for j in range(28)] for i in range(28)]
                    resultText, certaintyText = predict(canvas)
                    dataIndex += 1
                elif event.key == pygame.K_r:
                    canvas = [[0 for i in range(28)] for j in range(28)]
                    
        resultText, certaintyText = predict(canvas)

        mousePos = pygame.mouse.get_pos()

        if in_canvas(mousePos, drawingSurface, drawingSize):
            if buttons[1] and not buttons[3]:
                i, j = get_canvas_index(mousePos, drawingSurface, drawingSize)
                draw(canvas, (i, j))
            elif buttons[3] and not buttons[1]:
                i, j = get_canvas_index(mousePos, drawingSurface, drawingSize)
                erase(canvas, (i, j))

        win.fill((150, 150, 150))
        drawingSurface.fill((255, 255, 255))
        draw_canvas(canvas, drawingSurface, drawingSize)
        win.blit(drawingSurface, ((WIN_HEIGHT // 2) - (drawingSurface.get_height() // 2), (WIN_HEIGHT // 2) - (drawingSurface.get_height() // 2)))
        win.blit(resultText, (600, 100))
        win.blit(certaintyText, (600, 150))
        pygame.display.update()