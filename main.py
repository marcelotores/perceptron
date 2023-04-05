import numpy as np
import Perceptron

dataset = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


rotulos_ou = np.array([0, 1, 1, 1])
rotulos_and = np.array([0, 0, 0, 1])

pnn = Perceptron.Perceptron(0.5, 4)

print(pnn.treino2(dataset, rotulos_ou))


