import numpy as np


class Perceptron:

    def __init__(self, taxa_aprendizado=0.1, epocas=10):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas

    def treino(self, dataset, rotulos):
        entradas = dataset.shape[1]
        self.pesos = np.zeros(entradas)

        dataset_e_rotulos = zip(dataset, rotulos)

        for _ in range(self.epocas):
            for input, target in (dataset_e_rotulos):
                pass
                # novo_peso = peso + taxa_aprendizado * erro * input


    def step(self, x):
        return 1 if x > 0 else 0

    def predicao(self, data_set, pesos):
        dataset_predito = np.dot(data_set, pesos)
        return dataset_predito
        # peso_1 * entrada_1 +
        # peso_2 * entrada_2 +
        # peso_n * entrada_n + bias
        # chama a step e retorna ela
