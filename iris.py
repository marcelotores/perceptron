import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import seaborn
import mlp4

import mlpf
seaborn.set(style='whitegrid'); seaborn.set_context('talk')
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
from sklearn.datasets import load_iris
iris_data = load_iris()

n_samples, n_features = iris_data.data.shape
def Show_Diagram(_x ,_y,title):
    plt.figure(figsize=(10,4))
    plt.scatter(iris_data.data[:,_x],
    iris_data.data[:, _y], c=iris_data.target, cmap=cm.viridis)
    plt.xlabel(iris_data.feature_names[_x]);
    plt.ylabel(iris_data.feature_names[_y]);
    plt.title(title)
    plt.colorbar(ticks=([0, 1, 2]));
    plt.show();
#Show_Diagram(0,1,'Sepal')
#Show_Diagram(2,3,'Petal')

random.seed(123)


def separate_data():
    part_A_train = iris_dataset[0:40]
    part_A_test = iris_dataset[40:50]
    part_B_train = iris_dataset[50:90]
    part_B_test = iris_dataset[90:100]
    part_C_train = iris_dataset[100:140]
    part_C_test = iris_dataset[140:150]
    train = np.concatenate((part_A_train,
                            part_B_train,
                            part_C_train))
    test = np.concatenate((part_A_test,
                           part_B_test,
                           part_C_test))
    return [train, test]


train_porc = 80  # Porcent Training
test_porc = 20  # Porcent Test
# Join X and Y
iris_dataset = np.column_stack((iris_data.data,
                                iris_data.target.T))
iris_dataset = list(iris_dataset)
random.shuffle(iris_dataset)
file_train, file_test = separate_data()
train_X = np.array([i[:4] for i in file_train])
train_y = np.array([i[4] for i in file_train])
test_X = np.array([i[:4] for i in file_test])
test_y = np.array([i[4] for i in file_test])

train_y = train_y.reshape((120, 1))
print(train_X.shape)
print(train_y.shape)

## Parâmetros
taxa_aprendizado = 0.1
epocas = 1000
qtd_neuronios_camada_oculta = 3

qtd_neuronios_camada_saida = 3

## Definição de parâmetros
#mlp = mlpf.Mlp(train_X, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)

## Treino
#errors, param = mlp.treino(train_X, train_y)

## Teste
#y_pred = mlp.predicao(train_X, param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])

## Cálculo de acurácia
#num_correct_predictions = (y_pred == y).sum()
#accuracy = (num_correct_predictions / y.shape[0]) * 100
#print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)

##############

errors, param = mlp4.fit(train_all, rotulo_all, n_features=4, n_neurons=4, n_output=3, iterations=5000, eta=0.1)
# y_pred = mlp4.predict(test_all, param["W1"], param["W2"], param["b1"], param["b2"])
# num_correct_predictions = (y_pred == rotulo_test_all).sum()
# accuracy = (num_correct_predictions / y.shape[0]) * 100
# print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)