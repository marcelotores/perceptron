import math
import numpy as np
import cmath
import mlp4
from sklearn.datasets import load_iris


# expected values
y = np.array([[0, 1, 1, 0]]).T
print(y.shape)

# features
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]).T
print(X.shape)
#errors, param = mlp4.fit(X, y, iterations=5000, eta=0.1)

#y_pred = mlp4.predict(X, param["W1"], param["W2"], param["b1"], param["b2"])
#num_correct_predictions = (y_pred == y).sum()
#accuracy = (num_correct_predictions / y.shape[0]) * 100
#print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)


iris = load_iris()
X = iris['data']
y = iris['target'].reshape((150, 1))
X = X.T
y = y.T
print(X.shape)
print(y.shape)
errors, param = mlp4.fit(X, y, n_neurons=4, n_output=3, iterations=5000, eta=0.1)

#def fit(X, y, n_features=2, n_neurons=3, n_output=1, iterations=10, eta=0.001):

#y_pred = mlp4.predict(X, param["W1"], param["W2"], param["b1"], param["b2"])
#num_correct_predictions = (y_pred == y).sum()
#accuracy = (num_correct_predictions / y.shape[0]) * 100
#print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)

