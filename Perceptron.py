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
            print(f"--- Epoca {self.epocas} ---")
            # print(f"Pesos: {self.pesos}")
            count = 0
            for input, target in zip(dataset, rotulos):
                vetor_soma_pesos_1 = self.predicao(input, self.pesos1, self.bias_camada_oculta)
                predicoes1 = self.sigmoide(vetor_soma_pesos_1)
                vetor_soma_pesos_2 = self.predicao(predicoes1, self.pesos2, self.bias_camada_saida)
                #predicoes2 = self.step(vetor_soma_pesos_2)
                predicoes2 = self.sigmoide(vetor_soma_pesos_2)

                print(f'Entrada={input}, ground-truth={target}, pred={predicoes2}')

                # Contador para contabilizar o número de predições corretas

                for pred2 in predicoes2:
                    if pred2 != target:
                        erro = target - pred2
                        self.pesos2 += self.taxa_aprendizado * erro * predicoes1
                        #for pred1 in predicoes1:
                        self.pesos1 += self.taxa_aprendizado * erro * input
                    else:
                        count+=1
            acuracia = (count / dataset.shape[0] * 100)
            print(f'Acurácia: {acuracia}%')

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
        S2 = 1 / (1 + np.exp(-pesos1))
        return np.where(S2 >= 0.5, 1, 0)

    def predicao(self, input, pesos1, bias):
        vetor_funcao_soma_pesos_1 = np.zeros(0)
        for p1 in pesos1:
            funcao_soma_pesos = np.dot(input, p1)
            vetor_funcao_soma_pesos_1 = np.append(vetor_funcao_soma_pesos_1, funcao_soma_pesos)

        return vetor_funcao_soma_pesos_1 + bias
        # deve retornar um vetor de pesos



