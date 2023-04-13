import math

import numpy as np
import cmath
dataset = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
rotulos = np.array([0, 1, 1, 1])

dataset_e_rotulos = zip(dataset, rotulos)
dataset = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


def linear_function(W, dataset, b):
    """computes net input as dot product

    Args:
        W (ndarray): weight matrix
        X (ndarray): matrix of features
        b (ndarray): vector of biases

    Returns:
        Z (ndarray): weighted sum of features
        """

    return (X @ W) + b