import numpy as np
import mlpf

################################## Xor ####################################

## Rótulos
y = np.array([[0, 1, 1, 0]]).T


## Dataset

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]).T

## Parâmetros
taxa_aprendizado = 0.1
epocas = 10000
qtd_neuronios_camada_oculta = 4

qtd_neuronios_camada_saida = 1

## Definição de parâmetros
mlp = mlpf.Mlp(X, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)

## Treino
errors, param = mlp.treino(X, y)

## Teste
y_predicao = mlp.predicao(X, param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])

## Cálculo de acurácia
num_predicoes_corretas = (y_predicao == y).sum()

print('y_predicao', y_predicao)
print('rotulos', y)
print('num_predicoes_corretas', num_predicoes_corretas)

accuracy = (num_predicoes_corretas / y.shape[0]) * 100
print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)