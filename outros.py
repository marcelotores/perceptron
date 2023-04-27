import numpy as np


def sigmoide(soma_dos_pesos):
    return 1 / (1 + np.exp(-soma_dos_pesos))

# test_y = np.array([0.98, 0.90, 0.70, 0.60, -0.80, 0, 0.7, -0.50, -0.4]).reshape((3, 3))
#
# print(test_y)
# test_y = np.where(test_y <= -0.6, -1, np.where(test_y <= 0.6, 0, 1))
# print(test_y)
S = np.array([9.78250757, 9.78323099, 9.78204876, 9.78359854])
# 0.99994357 0.99994361 0.99994355 0.99994363
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

