import numpy as np


class Perceptron:

    def __init__(self, taxa_aprendizado=0.1, epocas=10):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas

    def treino(self, dataset, rotulos):
        entradas = dataset.shape[1]
        self.pesos = np.zeros(entradas)

        for _ in range(self.epocas):
            # peso_1 * entrada_1 +
            # peso_2 * entrada_2 +
            # peso_n * entrada_n + bias

    def step(self, x):
        return 1 if x > 0 else 0

    def predicao(data_set, pesos):
        pass
