import math
import numpy as np
import cmath
import mlp4
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlpf



# # expected values
y = np.array([[0, 1, 1, 0]]).T
#
#
# # features
X = np.array([[0, 0, 1, 1],
             [0, 1, 0, 1]]).T

mlp = mlpf.Mlp(X, 0.1, 4500, 3, 1)
errors, param = mlp.treino(X, y)
y_pred = mlp.predicao(X, param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])
num_correct_predictions = (y_pred == y).sum()
accuracy = (num_correct_predictions / y.shape[0]) * 100
print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)
#
#
#errors, param = mlp4.fit(X, y, n_features=2, iterations=5000, eta=0.1)
#y_pred = mlp4.predict(X, param["W1"], param["W2"], param["b1"], param["b2"])
#num_correct_predictions = (y_pred == y).sum()
#accuracy = (num_correct_predictions / y.shape[0]) * 100
#print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)
#print('Erro:', errors)


iris = load_iris()
X = iris['data']
y = iris['target'].reshape((150, 1))

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

#print('Dados de Teste: ', test_all)
#print('Rótulos de treino: ', rotulo_test_all)
#errors, param = mlp4.fit(train_all, rotulo_all, n_features=4, n_neurons=4, n_output=3, iterations=5000, eta=0.1)
#y_pred = mlp4.predict(test_all, param["W1"], param["W2"], param["b1"], param["b2"])
#num_correct_predictions = (y_pred == rotulo_test_all).sum()
#accuracy = (num_correct_predictions / y.shape[0]) * 100
#print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)

