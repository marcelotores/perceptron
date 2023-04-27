import pandas as pd
import mlpf
import numpy as np
import ut

## Importandno dados (numpy) por padrão. Para dataframe, use (data_frame=True) com segundo parâmetro
dataSet = ut.im_data(4)

## Divindo os dados em treino e teste
## Retorna (numpy) ou (dataframe). Dependendo dos dados passados
treino, teste = ut.divide_dados_treino_teste(dataSet, 0.8)

print(teste)






