import numpy as np


def sigmoide(soma_dos_pesos):
    return 1 / (1 + np.exp(-soma_dos_pesos))

# test_y = np.array([0.98, 0.90, 0.70, 0.60, -0.80, 0, 0.7, -0.50, -0.4]).reshape((3, 3))
#
# print(test_y)
# test_y = np.where(test_y <= -0.6, -1, np.where(test_y <= 0.6, 0, 1))
# print(test_y)
S = np.array([0.45786936, 0.90537796, 0.5043019])
# 0.61250861 0.71205342 0.62346976
Z = sigmoide(S)
print(Z)

# import matplotlib.pyplot as plt
# import numpy as np
#
# xpoints = v
# ypoints = vta
#
# plt.plot(xpoints, ypoints, 'o')
# plt.show()

