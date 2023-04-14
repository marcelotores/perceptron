import numpy as np


class Perceptron:

    def __init__(self, taxa_aprendizado=0.1, epocas=10, camada_oculta=1, bias_camada_oculta=1, camada_saida=1, bias_camada_saida=1):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.camada_oculta = camada_oculta
        self.camada_saida = camada_saida
        self.bias_camada_oculta = bias_camada_oculta
        self.bias_camada_saida = bias_camada_saida

    def treino(self, dataset, rotulos):
        qtd_col_dataset = dataset.shape[1]
        self.pesos1 = np.zeros((self.camada_oculta, qtd_col_dataset))
        self.pesos2 = np.zeros((self.camada_saida, self.camada_oculta))
        print("[INFO] Treinando o perceptron")

        for _ in range(self.epocas):
            print(f"--- Epoca {_} ---")
            # print(f"Pesos: {self.pesos}")
            count = 0

            vetor_soma_pesos_1 = self.predicao(dataset, self.pesos1, self.bias_camada_oculta)
            predicoes1 = self.tangente_hiperbolica(vetor_soma_pesos_1)

            vetor_soma_pesos_2 = self.predicao(predicoes1, self.pesos2, self.bias_camada_saida)
            predicoes2 = self.tangente_hiperbolica(vetor_soma_pesos_2)

            #delta2 = (S2 - y) * S2 * (1 - S2)
            #W2_gradients = predicoes1.T @ delta2

            delta2 = (predicoes2 - rotulos) * (predicoes2 * (1 - predicoes2))
            W2_gradients = predicoes1.T @ delta2
            db2 = np.sum(delta2, axis=0, keepdims=True)
            #
            # # Atualizando os pesos da camada de saída e o bias
            self.pesos2 -= self.taxa_aprendizado * W2_gradients
            # self.bias_camada_saida -= self.taxa_aprendizado * db2
            #
            #
            # delta1 = np.dot(delta2, self.pesos2.T) * (predicoes1 * (1 - predicoes1))
            # dW1 = np.dot(input.T, delta1)
            # db1 = np.sum(delta1, axis=0, keepdims=True)
            # print('Delta1:', delta1)
            # print('Delta2', delta2)
            #
            # # Atualizando os pesos da camada oculta e o bias
            # self.pesos1 -= self.taxa_aprendizado * dW1
            # # self.bias_camada_oculta -= self.taxa_aprendizado * db1




    def step(self, pesos1):
        predicao = []
        for p1 in pesos1:
            if p1 > 0:
                predicao.append(1)
            else:
                predicao.append(0)
        arr = np.array(predicao)
        return arr

    def sigmoide(self, pesos1):
        return 1/(1+np.exp(-pesos1))

    def tangente_hiperbolica(self, pesos1):
        #return np.tanh(pesos1)
        S2 = (np.exp(pesos1) - np.exp(-pesos1)) / (np.exp(pesos1) + np.exp(-pesos1))
        return np.where(S2 >= 1, 1, 0)

    def predicao(self, input, pesos1, bias):
        return (input @ pesos1) + bias

    def predict(X, pesos1, pesos2, bias_camada_oculta, bias_camada_saida):
        """computes predictions with learned parameters

        Args:
            X (ndarray): matrix of features
            W1 (ndarray): weight matrix for the first layer
            W2 (ndarray): weight matrix for the second layer
            b1 (ndarray): bias vector for the first layer
            b2 (ndarray): bias vector for the second layer

        Returns:
            d (ndarray): vector of predicted values
        """

        # Z1 = self.predicao(pesos1, X, bias_camada_oculta)
        # S1 = self.predicao(pesos1)
        # Z2 = linear_function(pesos2, S1, bias_camada_saida)
        # S2 = sigmoid_function(Z2)
        return 1



