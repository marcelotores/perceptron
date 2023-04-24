import numpy as np

test_y = np.array([0.98, 0.90, 0.70, 0.60, -0.80, 0, 0.7, -0.50, -0.4]).reshape((3, 3))

print(test_y)
test_y = np.where(test_y <= -0.6, -1, np.where(test_y <= 0.6, 0, 1))
print(test_y)



