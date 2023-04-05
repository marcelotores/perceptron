import numpy as np


class Perceptron:

    def __init__(self, taxa_aprendizado=0.1, epocas=10):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas

    def treino(self, dataset, rotulos):
        entradas = dataset.shape[1]
        self.pesos = np.zeros(entradas)
        print(self.pesos)
        # dataset_e_rotulos = zip(dataset, rotulos)
        print("[INFO] Treinando o perceptron")
        for _ in range(self.epocas):
            print(f"--- Epoca {self.epocas} ---")
            # print(f"Pesos: {self.pesos}")
            for input, target in zip(dataset, rotulos):
                p = self.predicao(input, self.pesos)
                p_step = self.step(p)
                print(f'Entrada={input}, ground-truth={target}, pred={p_step}')
                print("pesos:", self.pesos)
                if target != p_step:
                    erro = target - p
                    self.pesos += self.taxa_aprendizado * erro * input
        # return self.pesos

    def step(self, x):
        return 1 if x > 0 else 0

    def predicao(self, input, pesos):
        soma_entradas_pesos = np.dot(input, pesos)
        bias = -1
        return soma_entradas_pesos + bias
