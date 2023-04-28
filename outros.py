import numpy as np


def tangente_hiperbolica(soma_dos_pesos):
    """Função tangente hiperbólica."""

    valor = (np.exp(soma_dos_pesos) - np.exp(-soma_dos_pesos)) / (np.exp(soma_dos_pesos) + np.exp(-soma_dos_pesos))
    return valor


S = np.array([0.45000000000003434343434343434949494994, 0.9, 50])
# 0.61250861 0.71205342 0.62346976
Z = tangente_hiperbolica(S)
print(Z)
