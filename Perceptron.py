import numpy as np


class Perceptron:

    def __init__(self, taxa_aprendizado=0.1, epocas=10, camada_oculta=1, camada_saida=1 ):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.camada_oculta = camada_oculta

    def treino(self, dataset, rotulos):
        entradas = dataset.shape[1]

        ## criando o vetor de pesos com base na entrada
        #self.pesos = np.zeros((self.camada_oculta, entradas))
        self.pesos = np.zeros(entradas)

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

    def treino2(self, dataset, rotulos):
        entradas = dataset.shape[1]

        ## criando o vetor de pesos com base na entrada
        #self.pesos = np.zeros((self.camada_oculta, entradas))
        self.pesos = np.zeros(entradas)

        print("[INFO] Treinando o perceptron")

        for _ in range(self.epocas):
            print(f"--- Epoca {self.epocas} ---")
            # print(f"Pesos: {self.pesos}")
            for input, target in zip(dataset, rotulos):
                # vetor de pesos somados
                p = self.predicao2(input, self.pesos)

                # vetor de valores preditos
                p_step = self.step2(p)

                #print(f'Entrada={input}, ground-truth={target}, pred={p_step}')
                print("pesos:", p)
                print("preditos: ", p_step)
                # for p_s in p_step:
                #     print('É diferente')
                    # if target != p_s:
                    #     erro = target - p
                    #     self.pesos += self.taxa_aprendizado * erro * input
    def step(self, x):
        return 1 if x > 0 else 0

    def step2(self, somas_pesos):
        #linhas = somas_pesos.shape[0]
        predicao = np.zeros(0)
        for p in somas_pesos:
            if p > 0:
                predicao = np.append(predicao, 1)
            else:
                predicao = np.append(predicao, 0)

        return predicao


    def predicao(self, input, pesos):
        soma_entradas_pesos = np.dot(input, pesos)
        bias = -1
        return soma_entradas_pesos + bias
        # deve retornar um vetor de pesos

    def predicao2(self, input, pesos):
        linhas = pesos.shape[0]
        somas_pesos = np.zeros(linhas)
        for p in pesos:
            soma_entradas_pesos = np.dot(input, p)
            bias = -1
            # ver bias, pois ele ta usando o mesmo
            somas_pesos = np.append(somas_pesos, soma_entradas_pesos + bias)
        return somas_pesos
        # deve retornar um vetor de pesos


