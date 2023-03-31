import numpy as np
import Perceptron

dataset = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

rotulos = np.array([0, 1, 1, 1])

pnn = Perceptron.Perceptron(0.5, 2)

pnn.treino(dataset, rotulos)
