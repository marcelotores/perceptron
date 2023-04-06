import numpy as np

dataset = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
rotulos = np.array([0, 1, 1, 1])

dataset_e_rotulos = zip(dataset, rotulos)

#print(tuple(dataset_e_rotulos))

# calcula tamanho do vetor
entradas = dataset.shape[1]
pesos2 = np.zeros((1, entradas))

pesos = np.zeros(entradas)

input = dataset[1]

#print("Multiplicação: ", np.dot(input, pesos))

linhas = pesos2.shape[0]
vetor_pesos = np.zeros(linhas)

#print(linhas)
#print(vetor_pesos)

vetor_pesos = np.append(vetor_pesos, 2)
#print(vetor_pesos)

#vetor_pesos = np.append(vetor_pesos, 5)
#print(vetor_pesos)

#print(vetor_pesos.shape[0])

bias_camada_oculta = np.array([1, 2])

vetor_soma = np.array(([1, 1]))
print(bias_camada_oculta + vetor_soma)