import mlpf
import ut
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport


## Importandno dados (numpy) por padrão. Para dataframe, use (data_frame=True) com segundo parâmetro
dataSet = ut.im_data(4)
dataSet_3_classes = dataSet[:315, :]

# Dividindo os dados em 2 classes
classe1 = dataSet[:82, :]
classe2 = dataSet[315:, :]
classe1_classe2 = np.concatenate((classe1, classe2), axis=0)

treino2, teste2 = ut.divide_dados_treino_teste(classe1_classe2, 0.7)
print(treino2.shape)
print(teste2.shape)

#dados_df = pd.read_csv('dados_classificacao2.csv')



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

y_teste = ut.converte_rotulo_3(y_teste)
y_treino = ut.converte_rotulo_3(y_treino)

#print(ut.numero_atributo_por_classe())
#print(y_treino)
#print(y_teste)

## Parâmetros da Rede
taxa_aprendizado = 0.1
epocas = 1
qtd_neuronios_camada_oculta = 2
qtd_neuronios_camada_saida = 1

## Definição de parâmetros
mlp = mlpf.Mlp(treino2[:, :24], taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)

#X_treino[:10,:]
#y_treino[:10,:]
## Treino
#errors, param = mlp.treino(treino2[:, :24], treino2[:, 24].reshape(treino2.shape[0], 1))
#print(f'pesos camada oculta: {param["pesos_camada_oculta"]}')
#print(f'pesos camada saída: {param["pesos_camada_saida"]}')

#y_predicao = mlp.predicao(teste2[:, :24], param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])
#print(y_predicao)
## Cálculo de acurácia
#num_predicoes_corretas = (y_predicao == y_teste).sum()

#acuracia = (num_predicoes_corretas / y_teste.shape[0]) * 100
#print('Acurácia: %.2f%%' % acuracia)

#print(errors)

# Gráfico de Erro
#uti.grafico_erro(errors)


