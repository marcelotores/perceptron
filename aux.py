import numpy as np
# Funciona
#errors, param = mlp4.fit(train_X, train_y, n_features=4, n_neurons=4, n_output=3, iterations=5000, eta=0.1)
#y_pred = mlp4.predict(train_X, param["W1"], param["W2"], param["b1"], param["b2"])
#num_correct_predictions = (y_pred == train_y).sum()
#accuracy = (num_correct_predictions / train_y.shape[0]) * 100
#print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)

y_predicao = np.array([1, 2, 3, 4])
train_y = np.array([1, 2, 4, 4])
num_predicoes_corretas = (y_predicao == train_y).sum()
print(num_predicoes_corretas)