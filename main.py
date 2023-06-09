import numpy as np
import Perceptron
import mlp
import mlp4

dataset = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# expected values
y = np.array([[0, 1, 1, 0]]).T

# features
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]).T

rotulos_ou = np.array([0, 1, 1, 1])
rotulos_and = np.array([0, 0, 0, 1])
rotulos_xor = np.array([0, 1, 1, 0])

#bias_camada_oculta = np.array([-1, 1.5])
bias_camada_oculta = np.array([-1, 1])
bias_camada_saida = np.array([1])
neuronios_camada_oculta = 2
neuronios_camada_saida = 1
epocas = 1000
taxa_de_aprendizado = 0.1

#pnn = Perceptron.Perceptron(taxa_de_aprendizado, epocas, neuronios_camada_oculta, bias_camada_oculta, neuronios_camada_saida, bias_camada_saida)

#pnn.treino(dataset, rotulos_ou)


