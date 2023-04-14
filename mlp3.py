import numpy as np


class MLP:
    def _init_(self, input_size, hidden_size, output_size):
        """Inicializa a MLP com os tamanhos de entrada, camada oculta e saída."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicializa os pesos aleatoriamente
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros(self.output_size)

    def forward(self, X):
        """Realiza a propagação para a frente da MLP."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.tanh(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        """Realiza a retropropagação para atualização dos pesos."""
        m = X.shape[0]  # número de exemplos de treinamento

        # Calcula os gradientes
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        dZ1 = np.dot(dZ2, self.W2.T) * (1 - np.tanh(self.z1) ** 2)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m

        # Atualiza os pesos
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        """Treina a MLP usando o algoritmo de Gradiente Descendente."""
        for epoch in range(epochs):
            # Propagação para a frente
            output = self.forward(X)

            # Retropropagação e atualização dos pesos
            self.backward(X, y, learning_rate)

            # Cálculo do custo (erro)
            cost = np.mean((output - y) ** 2)

            # Imprime o custo a cada 100 épocas
            if epoch % 100 == 0:
                print("Época {}: Custo = {}".format(epoch, cost))

    def predict(self, X):
        """Realiza a predição com a MLP treinada."""
        return self.forward(X)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

mlp = MLP(2, 2, 1)

mlp.train(X, y, 100, 0.1)


#errors, param = (X, y, iterations=5000, eta=0.1)

#y_pred = predict(X, param["W1"], param["W2"], param["b1"], param["b2"])
#num_correct_predictions = (y_pred == y).sum()
#accuracy = (num_correct_predictions / y.shape[0]) * 100
#print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)