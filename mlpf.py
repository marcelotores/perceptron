import numpy as np
class Mlp():
    def __init__(self, dataset, taxa_aprendizado=0.1, epocas=10, qtd_neuronios_camada_oculta=1, qtd_neuronios_camada_saida=1):

        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.qtd_neuronios_camada_oculta = qtd_neuronios_camada_oculta
        self.qtd_neuronios_camada_saida = qtd_neuronios_camada_saida

        qtd_col_dataset = dataset.shape[1]
        self.pesos_camada_1 = np.zeros((qtd_col_dataset, self.qtd_neuronios_camada_oculta), dtype=np.float128)
        #self.pesos_camada_1 = np.random.uniform(size=(qtd_col_dataset,  self.qtd_neuronios_camada_oculta))
        self.bias_camada_oculta = np.random.uniform(size=(1, self.qtd_neuronios_camada_oculta))
        self.pesos_camada_2 = np.zeros((self.qtd_neuronios_camada_oculta, self.qtd_neuronios_camada_saida), dtype=np.float128)
        #self.pesos_camada_2 = np.random.uniform(size=(self.qtd_neuronios_camada_oculta, self.qtd_neuronios_camada_saida))
        self.bias_camada_saida = np.random.uniform(size=(1, self.qtd_neuronios_camada_saida))

    def funcao_linear(self, pesos, dataset, bias):
        return np.dot(dataset, pesos) + bias

    def sigmoide(self, soma_dos_pesos):
        return 1 / (1 + np.exp(-soma_dos_pesos))

    def step(self, pesos1):
        predicao = []
        for p1 in pesos1:
            if p1 > 0:
                predicao.append(1)
            else:
                predicao.append(0)
        arr = np.array(predicao)
        return arr

    def tangente_hiperbolica(self, soma_dos_pesos):
        """Função tangente hiperbólica."""

        return (np.exp(soma_dos_pesos) - np.exp(-soma_dos_pesos)) / (np.exp(soma_dos_pesos) + np.exp(-soma_dos_pesos))

    def custo(self, neuronios_ativados, rotulos):
        return (np.mean(np.power(neuronios_ativados - rotulos, 2))) / 2

    def predicao(self, dataset, pesos_camada_1, pesos_camada_2, bias_camada_oculta, bias_camada_saida):

        Z1 = self.funcao_linear(pesos_camada_1, dataset, bias_camada_oculta)
        #S1 = self.step(Z1)
        #S1 = self.sigmoide(Z1)
        S1 = self.tangente_hiperbolica(Z1)
        Z2 = self.funcao_linear(pesos_camada_2, S1, bias_camada_saida)
        #S2 = self.step(Z2)
        S2 = self.tangente_hiperbolica(Z2)
        #S2 = self.sigmoide(Z2)
        #return np.where(S2 >= 0.5, 1, 0)
        return np.where(S2 <= -0.6, -1, np.where(S2 <= 0.6, 0, 1))

    def treino(self, X, y):
        ## ~~ Initialize parameters ~~##

        ## ~~ storage errors after each iteration ~~##
        errors = []

        for _ in range(self.epocas):
            print(f'Época {_}')
            #print(self.pesos_camada_1)
            #print(self.pesos_camada_2)
            ## Forward ##

            Z1 = self.funcao_linear(self.pesos_camada_1, X, self.bias_camada_oculta)
            #S1 = self.tangente_hiperbolica(Z1)
            S1 = self.sigmoide(Z1)
            Z2 = self.funcao_linear(self.pesos_camada_2, S1, self.bias_camada_saida)
            #S2 = self.tangente_hiperbolica(Z2)
            S2 = self.sigmoide(Z2)

            ## Erros ##
            error = self.custo(S2, y)
            errors.append(error)

            ## Calcula os Gradientes ##

            delta2 = (S2 - y) * (S2 * (1 - S2))
            gradiente_peso2 = np.dot(S1.T, delta2)
            db2 = np.sum(delta2, axis=0)

            delta1 = np.dot(delta2, self.pesos_camada_2.T) * (S1 * (1 - S1))
            gradiente_peso1 = np.dot(X.T, delta1)
            db1 = np.sum(delta1, axis=0)

            # Atualização dos pesos
            self.pesos_camada_2 -= self.taxa_aprendizado * gradiente_peso2
            self.bias_camada_saida -= self.taxa_aprendizado * db2
            self.pesos_camada_1 -= self.taxa_aprendizado * gradiente_peso1
            self.bias_camada_oculta -= self.taxa_aprendizado * db1

            print('Z2', Z2)
            print('S2', S2)
            #print(y)
            #print('Erro: ', error)

            parametros = {
                "pesos_camada_oculta": self.pesos_camada_1,
                "pesos_camada_saida": self.pesos_camada_2,
                "bias_camada_oculta": self.bias_camada_oculta,
                "bias_camada_saida": self.bias_camada_saida
            }

        return errors, parametros
