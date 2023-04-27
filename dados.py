import mlpf
import ut
import numpy as np

## Importandno dados (numpy) por padrão. Para dataframe, use (data_frame=True) com segundo parâmetro
dataSet = ut.im_data(4)
dataSet_3_classes = dataSet[:315, :]

## Divindo os dados em treino e teste
## Retorna (numpy) ou (dataframe). Dependendo dos dados passados
treino, teste = ut.divide_dados_treino_teste(dataSet_3_classes, 0.7)


## separando rótulos do dataset para 3 classes
X_treino = treino[:, :24]
y_treino = treino[:, 24].reshape(treino.shape[0], 1)
X_teste = teste[:, :24]
y_teste = teste[:, 24].reshape(teste.shape[0], 1)


## separando rótulos do dataset para 4 classes
#X_treino = treino[:, :24]
#y_treino = treino[:, 24].reshape(300, 1)
# X_teste = teste[:, :24]
# y_teste = teste[:, 24].reshape(75, 1)

#print(X_treino)
#print(y_treino)
# print(X_teste.shape)
# print(y_teste.shape)


## Parâmetros da Rede
taxa_aprendizado = 0.01
epocas = 1
qtd_neuronios_camada_oculta = 3
qtd_neuronios_camada_saida = 4

## Definição de parâmetros
mlp = mlpf.Mlp(X_treino, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)

## Treino
errors, param = mlp.treino(X_treino, y_treino)

y_predicao = mlp.predicao(X_teste, param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])
print(y_predicao)
## Cálculo de acurácia
num_predicoes_corretas = (y_predicao == y_teste).sum()

acuracia = (num_predicoes_corretas / y_teste.shape[0]) * 100
print('Acurácia: %.2f%%' % acuracia)

print(errors)





