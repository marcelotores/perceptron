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

    def treino(self, dataset, rotulos):

        qtd_col_dataset = dataset.shape[1]
        self.pesos1 = np.zeros((self.camada_oculta, qtd_col_dataset))
        self.pesos2 = np.zeros((self.camada_saida, self.camada_oculta))
        print("[INFO] Treinando o perceptron")



        for _ in range(self.epocas):
            print(f"--- Epoca {_} ---")
            # print(f"Pesos: {self.pesos}")
            count = 0
            for input, target in zip(dataset, rotulos):
                vetor_soma_pesos_1 = self.predicao(input, self.pesos1, self.bias_camada_oculta)
                predicoes1 = self.sigmoide(vetor_soma_pesos_1)
                vetor_soma_pesos_2 = self.predicao(predicoes1, self.pesos2, self.bias_camada_saida)
                # predicoes2 = self.step(vetor_soma_pesos_2)
                predicoes2 = self.sigmoide(vetor_soma_pesos_2)
                #predicoes2 = self.hiperbolica2(vetor_soma_pesos_2)

                print(f'Entrada={input}, ground-truth={target}, pred={predicoes2}')

                # Contador para contabilizar o número de predições corretas

                delta2 = (predicoes2 - target) * predicoes2 * (1 - predicoes2)
                W2_gradients = predicoes1.T @ delta2
                self.pesos2 = self.pesos2 - W2_gradients * self.taxa_aprendizado

                # update output bias
                self.bias_camada_saida = self.bias_camada_saida - np.sum(delta2, axis=0, keepdims=True) * self.taxa_aprendizado

                # update hidden weights
                delta1 = (delta2 @ self.pesos2) * predicoes1 * (1 - predicoes1)
                W1_gradients = dataset.T @ delta1
                self.pesos1 = self.pesos1 - W1_gradients * self.taxa_aprendizado

                # update hidden bias
                self.bias_camada_oculta = self.bias_camada_oculta - np.sum(delta1, axis=0, keepdims=True) * self.taxa_aprendizado

#                for pred2 in predicoes2:
#                    if pred2 != target:
#                        erro = target - pred2
#                        self.pesos2 += self.taxa_aprendizado * erro * predicoes1
#                        for pred1 in predicoes1:
#                            self.pesos1 += self.taxa_aprendizado * erro * input
#                    else:
#                        count += 1
#            acuracia = (count / dataset.shape[0] * 100)
#            print(f'Acurácia: {acuracia}%')
