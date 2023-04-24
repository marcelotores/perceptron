import math
import numpy as np
import cmath
import mlp4
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlpf

################################## Xor ####################################

# ## Rótulos
# y = np.array([[0, 1, 1, 0]]).T
#
#
# ## Dataset
# X = np.array([[0, 0, 1, 1],
#              [0, 1, 0, 1]]).T
#
# ## Parâmetros
# taxa_aprendizado = 0.1
# epocas = 4500
# qtd_neuronios_camada_oculta = 3
#
# qtd_neuronios_camada_saida = 1
#
# ## Definição de parâmetros
# mlp = mlpf.Mlp(X, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)
#
# ## Treino
# errors, param = mlp.treino(X, y)
#
# ## Teste
# y_pred = mlp.predicao(X, param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])
#
# ## Cálculo de acurácia
# num_correct_predictions = (y_pred == y).sum()
# accuracy = (num_correct_predictions / y.shape[0]) * 100
# print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)


################################## Íris ####################################
#
iris = load_iris()
X = iris['data']
y = iris['target'].reshape((150, 1))

print(X.shape)
print(y.shape)
#
# ## Parâmetros
# taxa_aprendizado = 0.1
# epocas = 4500
# qtd_neuronios_camada_oculta = 3
#
# qtd_neuronios_camada_saida = 3
#
# ## Definição de parâmetros
# mlp = mlpf.Mlp(X, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)
#
# ## Treino
# errors, param = mlp.treino(X, y)
#
# ## Teste
# y_pred = mlp.predicao(X, param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])
#
# ## Cálculo de acurácia
# num_correct_predictions = (y_pred == y).sum()
# accuracy = (num_correct_predictions / y.shape[0]) * 100
# print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)
########################################################3333

classe_0 = X[0:50]
rotulo_0 = y[0:50]
classe_1 = X[50:100]
rotulo_1 = y[50:100]
classe_2 = X[100:150]
rotulo_2 = y[100:150]


train_classe_0, test_classe_0 = train_test_split(classe_0, test_size=10, random_state=42)
train_classe_0_rotulo = rotulo_0[:40]
test_classe_0_rotulo = rotulo_0[40:50]
#
train_classe_1, test_classe_1 = train_test_split(classe_1, test_size=10, random_state=42)
train_classe_1_rotulo = rotulo_1[:40]
test_classe_1_rotulo = rotulo_1[40:50]
#
train_classe_2, test_classe_2 = train_test_split(classe_2, test_size=10, random_state=42)
train_classe_2_rotulo = rotulo_2[:40]
test_classe_2_rotulo = rotulo_2[40:50]

## Dados de Treinamento
train_all = np.concatenate((train_classe_0, train_classe_1, train_classe_2))
rotulo_all = np.concatenate((train_classe_0_rotulo, train_classe_1_rotulo, train_classe_2_rotulo))
#print('Dados de treino: ', train_all)
#print('Rótulos de treino: ', rotulo_all)

## Dados de Teste
test_all = np.concatenate((test_classe_0, test_classe_1, test_classe_2))
rotulo_test_all = np.concatenate((test_classe_0_rotulo, test_classe_1_rotulo, test_classe_2_rotulo))
#
# #print('Dados de Teste: ', test_all)
# #print('Rótulos de treino: ', rotulo_test_all)


# errors, param = mlp4.fit(train_all, rotulo_all, n_features=4, n_neurons=4, n_output=3, iterations=5000, eta=0.1)
# y_pred = mlp4.predict(test_all, param["W1"], param["W2"], param["b1"], param["b2"])
# num_correct_predictions = (y_pred == rotulo_test_all).sum()
# accuracy = (num_correct_predictions / y.shape[0]) * 100
# print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)

print(X)

