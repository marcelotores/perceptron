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
            for input, target in zip(dataset, rotulos):
                vetor_soma_pesos_1 = self.predicao(input, self.pesos1, self.bias_camada_oculta)
                predicoes1 = self.step(vetor_soma_pesos_1)
                vetor_soma_pesos_2 = self.predicao(predicoes1, self.pesos2, self.bias_camada_saida)
                predicoes2 = self.step(vetor_soma_pesos_2)

                print(f'Entrada={input}, ground-truth={target}, pred={predicoes2}')

                for pred2 in predicoes2:
                    if pred2 != target:
                        erro = target - pred2
                        self.pesos2 += self.taxa_aprendizado * erro * predicoes1
                        for pred1 in predicoes1:
                            self.pesos1 += self.taxa_aprendizado * erro * input

            #print('vetor soma de pesos1: ', vetor_soma_pesos_1)
            #print('vetor de pesos1: ', self.pesos1)
            #print('vetor de predições1: ', predicoes1)
            #print('vetor soma de pesos2: ', vetor_soma_pesos_2)
            #print('vetor de pesos 2: ', self.pesos2)
            #print('vetor de predições2: ', predicoes2)
                #print('predicao:', predicoes1)
                #p_step = self.step(p)
                #print(f'Entrada={input}, ground-truth={target}, pred={p_step}')
                #print("pesos:", self.pesos)
                #if target != p_step:
                    #erro = target - p
                    #self.pesos += self.taxa_aprendizado * erro * input
        # return self.pesos

    def step(self, pesos1):
        predicao = []
        for p1 in pesos1:
            if p1 > 0:
                predicao.append(1)
            else:
                predicao.append(0)
        arr = np.array(predicao)
        return arr

    def predicao(self, input, pesos1, bias):
        vetor_funcao_soma_pesos_1 = np.zeros(0)
        for p1 in pesos1:
            funcao_soma_pesos = np.dot(input, p1)
            vetor_funcao_soma_pesos_1 = np.append(vetor_funcao_soma_pesos_1, funcao_soma_pesos)

        return vetor_funcao_soma_pesos_1 + bias
        # deve retornar um vetor de pesos



