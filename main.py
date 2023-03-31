import numpy as np
import Perceptron

dataset = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

rotulos = np.array([0, 1, 1, 1])
pesos = np.zeros(2)
print(pesos)

pnn = Perceptron.Perceptron(0.5, 2)

#pnn.treino(dataset, rotulos)
print(pnn.predicao(dataset, pesos))

