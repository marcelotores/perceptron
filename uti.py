import numpy as np
from matplotlib import pyplot as plt


def str_to_numpy_4(dados):
    vetor_lista = []

    for i in dados:
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

    dados = np.array(vetor_lista)
    # yy = dados.shape = (375, 1)
    return dados.reshape((375, 1))

def str_to_numpy_10(dados):
    vetor_lista = []

    for i in dados:
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

    dados = np.array(vetor_lista)
    #yy = dados.shape = (375, 1)
    return dados.reshape((375, 1))

def grafico_erro(erros):
    loss_values = erros
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, label='Erro de Treinamento')
    plt.xlabel('Epocas')
    plt.ylabel('Erro')
    plt.legend()

    plt.show()

def converte_rotulo_3(y):
    novo_y_teste = []
    for i in y:
        if i == 1:
            novo_y_teste.append([1, -1, -1])
        elif i == 2:
            novo_y_teste.append([-1, 1, -1])
        else:
            novo_y_teste.append([-1, -1, 1])
    y_num = np.array(novo_y_teste)
    return y_num