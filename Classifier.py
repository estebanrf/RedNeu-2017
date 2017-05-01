from perceptronMulticapa import PerceptronMulticapa, binary_sigmoidal, binary_sigmoidal_derivative
from numpy import argmax, mean, std
from Parser import parse_ej1


class Classifier(PerceptronMulticapa):
    def __init__(self, *args):
        super(Classifier, self).__init__(*args)

    def out(self, X, Y, activation_fx):
        error = 0
        for a, EXPECTED in zip(X, Y):
            V = self.calculate_V(a, activation_fx)
            V_M = V[-1]

            index_max = argmax(V_M[-1])
            out = V_M[index_max]
            print "{} {}: {}".format(index_max, EXPECTED, out)

def normalize_minmax(X, Y):
    def normalize_list(V):
        # para cada parametro
        for param_num in range(0, len(V[0])):
            param_data = [pttrn[param_num] for pttrn in V]
            param_min = min(param_data)
            param_max = max(param_data)
            for pttrn in V:
                pttrn[param_num] = 2 * (pttrn[param_num] - param_min) / (param_max - param_min - 1)
    normalize_list(X)
    normalize_list(Y)

def normalize_standarize(X, Y):
    def normalize_list(V):
        # para cada parametro
        for param_num in range(0, len(V[0])):
            param_data = [pttrn[param_num] for pttrn in V]
            param_mean = mean(param_data)
            param_std = std(param_data)
            for pttrn in V:
                pttrn[param_num] = (pttrn[param_num] - param_mean) / param_std
    normalize_list(X)
    normalize_list(Y)

X, Y = parse_ej1()
normalize_standarize(X, Y)
hiddenLayers = [2,3] #neuron numbers for each hidden layer
mlp = Classifier(len(X[0]), hiddenLayers, len(Y[0]))
mlp.train(X, Y, binary_sigmoidal, binary_sigmoidal_derivative)
mlp.out(X, Y, binary_sigmoidal)