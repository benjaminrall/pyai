import pyai
import pygame
import random
import numpy as np

# Constants
WIN_WIDTH = 900
WIN_HEIGHT = 650
FRAMERATE = 240

# Loads the neural network for recognising handwritten digits
network = pyai.Network.load("mnist_network.pyai")

# Window setup
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
win.fill((150, 150, 150))
pygame.display.set_caption("Digit Recognition")

# Drawing surface setup
drawingSize = 18
drawingSurface = pygame.Surface((28 * drawingSize, 28 * drawingSize))

# Font setup
pygame.font.init()
font = pygame.font.SysFont("georgia", 30)
resultText = font.render("Result: ", True, (0, 0, 0))
certaintyText = font.render("Certainty: ", True, (0, 0, 0))

# Variable setup
running = True
canvas = [[0 for _ in range(28)] for _ in range(28)]
buttons = {1: False, 3: False}
dataIndex = 0

clock = pygame.time.Clock()

# Draws the canvas to a surface
def draw_canvas(canvas, surface, size):
    for r, row in enumerate(canvas):
        for c, col in enumerate(row):
            col = (-col) + 1
            colour = (255 * col, 255 * col, 255 * col)
            dimensions = (c * size, r * size, size, size)
            pygame.draw.rect(surface, colour, dimensions)

# Checks if a position lies within the canvas
def in_canvas(pos, surface, size):
    xmin = (WIN_HEIGHT // 2) - (surface.get_height() // 2)
    ymin = (WIN_HEIGHT // 2) - (surface.get_height() // 2)
    xmax = xmin + (size * 28)
    ymax = ymin + (size * 28)
    return pos[0] >= xmin and pos[0] < xmax and pos[1] >= ymin and pos[1] < ymax

# Gets the row and column index of the square in which the cursor lies
def get_canvas_index(pos, surface, size):
    xmin = (WIN_HEIGHT // 2) - (surface.get_height() // 2)
    ymin = (WIN_HEIGHT // 2) - (surface.get_height() // 2)
    pos = ((pos[0] - xmin) // size, (pos[1] - ymin) // size)
    return pos[1], pos[0]

# Draws to the canvas
def draw(canvas, indexes):
    for i in range(-1, 2):
        for j in range(-1, 2):
            row = indexes[0] + i
            col = indexes[1] + j
            if 0 <= row < 28 and 0 <= col < 28 and canvas[row][col] != 1:
                canvas[row][col] += (1 - canvas[row][col]) / 4
    canvas[indexes[0]][indexes[1]] = 1

# Erases from the canvas
def erase(canvas, indexes):
    for i in range(-1, 2):
        for j in range(-1, 2):
            row = indexes[0] + i
            col = indexes[1] + j
            if 0 <= row < 28 and 0 <= col < 28:
                canvas[row][col] -= canvas[row][col] / 2
    canvas[indexes[0]][indexes[1]] = 0

# Predicts the results of the image drawn in the canvas
def predict(canvas):
    values = network(np.array([canvas])).reshape((10,))
    #print(values)
    answer = np.argmax(values)
    #print(answer)
    resultText = font.render(f"Result: {answer}", True, (0, 0, 0))
    certaintyText = font.render(f"Certainty: {round(values[answer] * 100, 1)}%", True, (0, 0, 0))
    return resultText, certaintyText

# Main program loop
while running:
    
    clock.tick(FRAMERATE)

    # Loops through pygame window events
    for event in pygame.event.get():    
        if event.type == pygame.QUIT:   # Allows the user to quit the program with the quit button
            running = False
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:  # Checks for mouse buttons being pressed down
            if event.button in buttons:
                buttons[event.button] = True
        elif event.type == pygame.MOUSEBUTTONUP:    # Checks for mouse buttons being released
            if event.button in buttons:
                buttons[event.button] = False
        elif event.type == pygame.KEYDOWN:  # Checks for if a key has been pressed down
            if event.key == pygame.K_r:   # If 'R' is pressed, clear the canvas
                canvas = [[0 for _ in range(28)] for _ in range(28)]

    resultText, certaintyText = predict(canvas)

    mousePos = pygame.mouse.get_pos()

    # Handles drawing to the canvas
    if in_canvas(mousePos, drawingSurface, drawingSize):
        if buttons[1] and not buttons[3]:
            i, j = get_canvas_index(mousePos, drawingSurface, drawingSize)
            draw(canvas, (i, j))
        elif buttons[3] and not buttons[1]:
            i, j = get_canvas_index(mousePos, drawingSurface, drawingSize)
            erase(canvas, (i, j))

    # Updates all elements of the display
    win.fill((150, 150, 150))
    drawingSurface.fill((255, 255, 255))

    draw_canvas(canvas, drawingSurface, drawingSize)

    win.blit(drawingSurface, ((WIN_HEIGHT // 2) - (drawingSurface.get_height() // 2), (WIN_HEIGHT // 2) - (drawingSurface.get_height() // 2)))
    win.blit(resultText, (600, 100))
    win.blit(certaintyText, (600, 150))
        
    pygame.display.update()