import numpy as np
class Mlp():
    def __init__(self, dataset, taxa_aprendizado=0.1, epocas=10, qtd_neuronios_camada_oculta=1, qtd_neuronios_camada_saida=1):

        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.qtd_neuronios_camada_oculta = qtd_neuronios_camada_oculta
        self.qtd_neuronios_camada_saida = qtd_neuronios_camada_saida

        qtd_col_dataset = dataset.shape[1]
        self.pesos_camada_1 = np.random.uniform(size=(qtd_col_dataset,  self.qtd_neuronios_camada_oculta))
        self.bias_camada_oculta = np.random.uniform(size=(1, self.qtd_neuronios_camada_oculta))
        self.pesos_camada_2 = np.random.uniform(size=(self.qtd_neuronios_camada_oculta, self.qtd_neuronios_camada_saida))
        self.bias_camada_saida = np.random.uniform(size=(1, self.qtd_neuronios_camada_saida))

    def funcao_linear(self, pesos, dataset, bias):
        return np.dot(dataset, pesos) + bias

    def sigmoide(self, soma_dos_pesos):
        return 1 / (1 + np.exp(-soma_dos_pesos))

    def custo(self, neuronios_ativados, rotulos):
        return (np.mean(np.power(neuronios_ativados - rotulos, 2))) / 2

    def predicao(self, dataset, pesos_camada_1, pesos_camada_2, bias_camada_oculta, bias_camada_saida):

        Z1 = self.funcao_linear(pesos_camada_1, X, bias_camada_oculta)
        S1 = self.sigmoide(Z1)
        # S1 = tangente_hiperbolica(Z1)
        Z2 = self.funcao_linear(pesos_camada_2, S1, bias_camada_saida)
        # S2 = tangente_hiperbolica(Z2)
        S2 = self.sigmoide(Z2)
        return np.where(S2 >= 0.5, 1, 0)

    def treino(self, X, y):
        ## ~~ Initialize parameters ~~##

        ## ~~ storage errors after each iteration ~~##
        errors = []

        for _ in range(self.epocas):
            print(f'Época {_}')

            ## Forward ##

            Z1 = self.funcao_linear(self.pesos_camada_1, X, self.bias_camada_oculta)
            # S1 = tangente_hiperbolica(Z1)
            S1 = self.sigmoide(Z1)
            Z2 = self.funcao_linear(self.pesos_camada_2, S1, self.bias_camada_saida)
            # S2 = tangente_hiperbolica(Z2)
            S2 = self.sigmoide(Z2)

            ## Erros ##
            error = self.custo(S2, y)
            errors.append(error)

            ## Calcula os Gradientes ##

            delta2 = (S2 - y) * (S2 * (1 - S2))
            gradiente_peso2 = np.dot(S1.T, delta2)
            db2 = np.sum(delta2, axis=0)

            delta1 = np.dot(delta2, self.pesos_camada_2.T) * (S1 * (1 - S1))
            gradiente_peso1 = np.dot(X.T, delta1)
            db1 = np.sum(delta1, axis=0)

            # Atualização dos pesos
            self.pesos_camada_2 -= self.taxa_aprendizado * gradiente_peso2
            self.bias_camada_saida -= self.taxa_aprendizado * db2
            self.pesos_camada_1 -= self.taxa_aprendizado * gradiente_peso1
            self.bias_camada_oculta -= self.taxa_aprendizado * db1

            print('S1', S1)
            print('S2', S2)
            print('Erro: ', error)

            parametros = {
                "pesos_camada_oculta": self.pesos_camada_1,
                "pesos_camada_saida": self.pesos_camada_2,
                "bias_camada_oculta": self.bias_camada_oculta,
                "bias_camada_saida": self.bias_camada_saida
            }

        return errors, parametros
def init_parameters(n_features, n_neurons, n_output):
    """generate initial parameters sampled from an uniform distribution

    Args:
        n_features (int): number of feature vectors
        n_neurons (int): number of neurons in hidden layer
        n_output (int): number of output neurons

    Returns:
        parameters dictionary:
            W1: weight matrix, shape = [n_features, n_neurons]
            b1: bias vector, shape = [1, n_neurons]
            W2: weight matrix, shape = [n_neurons, n_output]
            b2: bias vector, shape = [1, n_output]
    """

    np.random.seed(100)  # for reproducibility
    W1 = np.random.uniform(size=(n_features, n_neurons))
    b1 = np.random.uniform(size=(1, n_neurons))
    W2 = np.random.uniform(size=(n_neurons, n_output))
    b2 = np.random.uniform(size=(1, n_output))
    # W1 = np.random.uniform(size=(n_neurons, n_features))
    # b1 = np.random.uniform(size=(1, n_neurons))
    # W2 = np.random.uniform(size=(n_output, n_neurons))
    # b2 = np.random.uniform(size=(1, n_output))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def linear_function(W, X, b):
    """computes net input as dot product

    Args:
        W (ndarray): weight matrix
        X (ndarray): matrix of features
        b (ndarray): vector of biases

    Returns:
        Z (ndarray): weighted sum of features
        """
    return np.dot(X, W) + b


def tangente_hiperbolica(Z):
    """Função tangente hiperbólica."""
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def sigmoid_function(Z):
    """computes sigmoid activation element wise

    Args:
        Z (ndarray): weighted sum of features

    Returns:
        S (ndarray): neuron activation
    """

    return 1 / (1 + np.exp(-Z))


def cost_function(A, y):
    """computes squared error

    Args:
        A (ndarray): neuron activation
        y (ndarray): vector of expected values

    Returns:
        E (float): total squared error"""

    return (np.mean(np.power(A - y, 2))) / 2


def predict(X, W1, W2, b1, b2):
    """computes predictions with learned parameters

    Args:
        X (ndarray): matrix of features
        W1 (ndarray): weight matrix for the first layer
        W2 (ndarray): weight matrix for the second layer
        b1 (ndarray): bias vector for the first layer
        b2 (ndarray): bias vector for the second layer

    Returns:
        d (ndarray): vector of predicted values
    """

    Z1 = linear_function(W1, X, b1)
    S1 = sigmoid_function(Z1)
    #S1 = tangente_hiperbolica(Z1)
    Z2 = linear_function(W2, S1, b2)
    #S2 = tangente_hiperbolica(Z2)
    S2 = sigmoid_function(Z2)
    return np.where(S2 >= 0.5, 1, 0)
    #return np.where(S2 >= 0, 1, 0)


def fit(X, y, n_features=2, n_neurons=3, n_output=1, iterations=10, eta=0.001):
    """Multi-layer perceptron trained with backpropagation

    Args:
        X (ndarray): matrix of features
        y (ndarray): vector of expected values
        n_features (int): number of feature vectors
        n_neurons (int): number of neurons in hidden layer
        n_output (int): number of output neurons
        iterations (int): number of iterations over the training set
        eta (float): learning rate

    Returns:
        errors (list): list of errors over iterations
        param (dic): dictionary of learned parameters
    """

    ## ~~ Initialize parameters ~~##
    param = init_parameters(n_features=n_features,
                            n_neurons=n_neurons,
                            n_output=n_output)

    ## ~~ storage errors after each iteration ~~##
    errors = []

    for _ in range(iterations):
        print(f'Época {_}')

        ## Forward ##

        Z1 = linear_function(param['W1'], X, param['b1'])
        #S1 = tangente_hiperbolica(Z1)
        S1 = sigmoid_function(Z1)
        Z2 = linear_function(param['W2'], S1, param['b2'])
        #S2 = tangente_hiperbolica(Z2)
        S2 = sigmoid_function(Z2)

        ## Erros ##
        error = cost_function(S2, y)
        errors.append(error)

        ## Calcula os Gradientes ##

        delta2 = (S2 - y) * (S2 * (1 - S2))
        gradiente_peso2 = np.dot(S1.T, delta2)
        db2 = np.sum(delta2, axis=0)

        delta1 = np.dot(delta2, param["W2"].T) * (S1 * (1 - S1))
        gradiente_peso1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        # Atualização dos pesos
        param["W2"] -= eta * gradiente_peso2
        param["b2"] -= eta * db2
        param["W1"] -= eta * gradiente_peso1
        param["b1"] -= eta * db1


        print('S1', S1)
        print('S2', S2)
        print('Erro: ', error)

    return errors, param

# expected values
y = np.array([[0, 1, 1, 0]]).T

# features
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]).T

#errors, param = fit(X, y, iterations=5000, eta=0.1)

#y_pred = predict(X, param["W1"], param["W2"], param["b1"], param["b2"])
#num_correct_predictions = (y_pred == y).sum()
#accuracy = (num_correct_predictions / y.shape[0]) * 100
#print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)