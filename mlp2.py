import numpy as np

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dados de entrada do problema do XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Função de treinamento da MLP
def train_mlp(X, y, hidden_units=2, learning_rate=0.1, epochs=1):
    # Inicialização dos pesos
    input_units = X.shape[1]
    output_units = 1
    W1 = np.random.rand(input_units, hidden_units)
    b1 = np.random.rand(hidden_units)
    W2 = np.random.rand(hidden_units, output_units)
    b2 = np.random.rand(output_units)

    # Loop de treinamento
    for epoch in range(epochs):
        # Forward pass
        a1 = np.dot(X, W1) + b1
        h1 = sigmoid(a1)
        a2 = np.dot(h1, W2) + b2
        h2 = sigmoid(a2)

        # Backward pass
        delta2 = (h2 - y) * (h2 * (1 - h2))
        dW2 = np.dot(h1.T, delta2)
        db2 = np.sum(delta2, axis=0)
        delta1 = np.dot(delta2, W2.T) * (h1 * (1 - h1))
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        # Atualização dos pesos
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1



    return W1, b1, W2, b2

# Treinamento da MLP
W1, b1, W2, b2 = train_mlp(X, y)
#print(W1,'\n\n',W2)

# Função de predição
def predict(X, W1, b1, W2, b2):
    a1 = np.dot(X, W1) + b1
    h1 = sigmoid(a1)
    a2 = np.dot(h1, W2) + b2
    h2 = sigmoid(a2)
    return np.round(h2).flatten()

# Predição com o modelo treinado
##preds = predict(X, W1, b1, W2, b2)

# Impressão dos resultados
#print("Dados de entrada: ", X)
#print("Valores reais: ", y)
#print("Valores preditos: ", preds)