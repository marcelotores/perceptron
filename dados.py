import pandas as pd
import mlpf
import numpy as np

dados_df = pd.read_csv('dados_classificacao2.csv')
#print(dados_df)
X = dados_df.iloc[:, :24].to_numpy()
y = dados_df.iloc[:, 24:].to_numpy()

vetor_lista = []

for i in y:
    if i == 'c1_p1':
        vetor_lista.append(1)
    elif i == 'c2_p1':
        vetor_lista.append(2)
    elif i == 'c3_p1':
        vetor_lista.append(3)
    elif i == 'c3_p2':
        vetor_lista.append(4)
    elif i == 'c3_p3':
        vetor_lista.append(5)
    elif i == 'c3_p4':
        vetor_lista.append(6)
    elif i == 'c4_p1':
        vetor_lista.append(7)
    elif i == 'c4_p1':
        vetor_lista.append(8)
    elif i == 'c4_p2':
        vetor_lista.append(9)
    elif i == 'c4_p3':
        vetor_lista.append(10)
    elif i == 'c4_p4':
        vetor_lista.append(11)


yy = np.array(vetor_lista)

yy.shape = (375, 1)
#print('esse ', yy)


numero_atributo_por_classe = dados_df.groupby(['classe'])['classe'].count()
#print((numero_atributo_por_classe))
#print(y.shape)

print(X[:1][:])
print(yy[:1][:])


numero_atributo_por_classe = dados_df.groupby(['classe'])['classe'].count()

taxa_aprendizado = 0.1
epocas = 4500
qtd_neuronios_camada_oculta = 3
qtd_neuronios_camada_saida = 11

mlp = mlpf.Mlp(X, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)
errors, param = mlp.treino(X, yy)
print(param["pesos_camada_oculta"].shape, param["pesos_camada_saida"].shape, param["bias_camada_oculta"].shape, param["bias_camada_saida"].shape)
y_pred = mlp.predicao(X, param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])
#num_correct_predictions = (y_pred == yy).sum()
#accuracy = (num_correct_predictions / yy.shape[0]) * 100
#print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)