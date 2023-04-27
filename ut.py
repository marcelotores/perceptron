import numpy as np
import pandas as pd

dados_df = pd.read_csv('dados_classificacao2.csv')

def numero_atributo_por_classe():
    return dados_df.groupby(['classe'])['classe'].count()

def im_data(qtd_cat=4, data_frame=False):
    #dados_df = pd.read_csv('dados_classificacao2.csv')
    X = dados_df.iloc[:, :24].to_numpy()
    y = dados_df.iloc[:, 24:].to_numpy()
    yy = str_to_numpy(y, qtd_cat)
    dataSet = np.concatenate((X, yy), axis=1)

    if data_frame:
        return pd.DataFrame(dataSet)
    return dataSet

def str_to_numpy(y, qtd_cat):
    vetor_lista = []

    if qtd_cat == 10:
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
            elif i == 'c4_p2':
                vetor_lista.append(8)
            elif i == 'c4_p3':
                vetor_lista.append(9)
            elif i == 'c4_p4':
                vetor_lista.append(10)
    elif qtd_cat == 4:
        for i in y:
            if i == 'c1_p1':
                vetor_lista.append(1)
            elif i == 'c2_p1':
                vetor_lista.append(2)
            elif i == 'c3_p1':
                vetor_lista.append(3)
            elif i == 'c3_p2':
                vetor_lista.append(3)
            elif i == 'c3_p3':
                vetor_lista.append(3)
            elif i == 'c3_p4':
                vetor_lista.append(3)
            elif i == 'c4_p1':
                vetor_lista.append(4)
            elif i == 'c4_p2':
                vetor_lista.append(4)
            elif i == 'c4_p3':
                vetor_lista.append(4)
            elif i == 'c4_p4':
                vetor_lista.append(4)


    yy = np.array(vetor_lista).reshape((375, 1))

    return yy

def divide_dados_treino_teste(dados, frac_treino):

    dados_df = pd.DataFrame(dados)

    training_data = dados_df.sample(frac=frac_treino, random_state=25)
    testing_data = dados_df.drop(training_data.index)

    #print(f"No. of training examples: {training_data.shape[0]}")
    #print(f"No. of testing examples: {testing_data.shape[0]}")

    if isinstance(dados, pd.DataFrame):
        return pd.DataFrame(training_data), pd.DataFrame(testing_data)

    return training_data.to_numpy(), testing_data.to_numpy()