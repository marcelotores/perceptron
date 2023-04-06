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
rotulos_xor = np.array([0, 1, 1, 0])

bias_camada_oculta = np.array([-1, 1.5])
bias_camada_saida = np.array([1.5])
neuronios_camada_oculta = 2
neuronios_camada_saida = 1
epocas = 100
taxa_de_aprendizado = 0.5

pnn = Perceptron.Perceptron(taxa_de_aprendizado, epocas, neuronios_camada_oculta, bias_camada_oculta, neuronios_camada_saida, bias_camada_saida)

print(pnn.treino(dataset, rotulos_xor))


