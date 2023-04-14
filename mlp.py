import cmath
import math

import numpy as np

class Mlp():

    def __init__(self,
                 taxa_aprendizado=0.1,
                 epocas=10,
                 camada_oculta=1,
                 bias_camada_oculta=1,
                 camada_saida=1,
                 bias_camada_saida=1):

        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.camada_oculta = camada_oculta
        self.camada_saida = camada_saida
        self.bias_camada_oculta = bias_camada_oculta
        self.bias_camada_saida = bias_camada_saida


    def step(self, pesos1):
        predicao = []
        for p1 in pesos1:
            if p1 > 0:
                predicao.append(1)
            else:
                predicao.append(0)
        arr = np.array(predicao)
        return arr

    def hiperbolica2(self, pesos1):
        predicao = []
        for p1 in pesos1:
            predicao.append(math.tanh(p1))
        arr = np.array(predicao)
        return arr

    def tangente_hiperbolica(self, pesos1):
        return np.tanh(pesos1)
        #return (np.exp(pesos1) - np.exp(-pesos1)) / (np.exp(pesos1) + np.exp(-pesos1))

    def sigmoide(self, pesos1):
        return 1 / (1 + np.exp(-pesos1))

    def predicao(self, input, pesos1, bias):
        vetor_funcao_soma_pesos_1 = np.zeros(0)
        for p1 in pesos1:
            funcao_soma_pesos = np.dot(input, p1)
            vetor_funcao_soma_pesos_1 = np.append(vetor_funcao_soma_pesos_1, funcao_soma_pesos)

        return vetor_funcao_soma_pesos_1 + bias
        # deve retornar um vetor de pesos

    def forward(self, dataset):
        """Realiza a propagação para a frente da MLP."""
        self.soma_pesos1 = np.dot(dataset, self.pesos1) + self.bias_camada_oculta
        self.funcao_ativacao1 = np.tanh(self.soma_pesos1)
        #self.soma_pesos2 = np.dot(self.funcao_ativacao1, self.pesos2) + self.bias_camada_saida
        #self.funcao_ativacao2 = np.tanh(self.soma_pesos2)
        #return self.funcao_ativacao2
        #return (dataset @ self.pesos1) + self.bias_camada_oculta
        return self.soma_pesos1
    def treino(self, dataset, rotulos):

        qtd_col_dataset = dataset.shape[1]
        self.pesos1 = np.zeros((self.camada_oculta, qtd_col_dataset))
        self.pesos2 = np.zeros((self.camada_saida, self.camada_oculta))
        print("[INFO] Treinando o perceptron")

        for _ in range(self.epocas):
            print(f"--- Epoca {_} ---")
            funcao_ativacao2 = self.forward(dataset)
            print(funcao_ativacao2)

