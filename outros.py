import numpy as np

# test_y = np.array([0.98, 0.90, 0.70, 0.60, -0.80, 0, 0.7, -0.50, -0.4]).reshape((3, 3))
#
# print(test_y)
# test_y = np.where(test_y <= -0.6, -1, np.where(test_y <= 0.6, 0, 1))
# print(test_y)

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)
def tangente_hiperbolica(soma_dos_pesos):
    """Função tangente hiperbólica."""

    return (np.exp(soma_dos_pesos) - np.exp(-soma_dos_pesos)) / (np.exp(soma_dos_pesos) + np.exp(-soma_dos_pesos))


val = np.array([8.69816126e-01, 5.52995792e-01, 4.10496011e-04])
print(val)
tanh = tangente_hiperbolica(val)
print(tanh)




# import matplotlib.pyplot as plt
# import numpy as np
#
# xpoints = v
# ypoints = vta
#
# plt.plot(xpoints, ypoints, 'o')
# plt.show()

