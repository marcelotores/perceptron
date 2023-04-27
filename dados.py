import mlpf
import ut
import numpy as np

## Importandno dados (numpy) por padrão. Para dataframe, use (data_frame=True) com segundo parâmetro
dataSet = ut.im_data(4)

## Divindo os dados em treino e teste
## Retorna (numpy) ou (dataframe). Dependendo dos dados passados
treino, teste = ut.divide_dados_treino_teste(dataSet, 0.8)

## separando rótulos do dataset
X_treino = treino[:, :24]
y_treino = treino[:, 24].reshape(300, 1)
X_teste = teste[:, :24]
y_teste = teste[:, 24].reshape(75, 1)

#print(X_treino)
#print(y_treino)
# print(X_teste.shape)
# print(y_teste.shape)


## Rótulos
y = np.array([[0, 1, 1, 0]]).T


## Dataset
X = np.array([[0, 0, 1, 1],
             [0, 1, 0, 1]]).T

## Parâmetros da Rede
taxa_aprendizado = 0.01
epocas = 10000
qtd_neuronios_camada_oculta = 3
qtd_neuronios_camada_saida = 4

## Definição de parâmetros
mlp = mlpf.Mlp(X_treino, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)

## Treino
errors, param = mlp.treino(X_treino, y_treino)






