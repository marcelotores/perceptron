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
rotulos3 = np.array([
    1, 2, 3
])

bias_camada_oculta = []
bias_camada_saida = []

pnn = Perceptron.Perceptron(0.5, 1, 2, -1, 3, 0)

print(pnn.treino(dataset, rotulos_ou))


