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

entradas = dataset.shape[1]
pesos = np.zeros(entradas)

print(entradas)
print(pesos)