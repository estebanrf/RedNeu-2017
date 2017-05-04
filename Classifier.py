from perceptronMulticapa import PerceptronMulticapa, Layer, TrainingSpecs, v_tanh, v_tanh_deriv
import matplotlib.pyplot as plt
from numpy import argmax, mean, std
from Parser import parse_ej1
import numpy as np

class Classifier(PerceptronMulticapa):
    def __init__(self, *args):
        super(Classifier, self).__init__(*args)

    def out(self, X, Y):
    #retorna errores/totales

        errors = 0
        for a, EXPECTED in zip(X, Y):
            V = self.calculate_V(a)
            V_M = V[-1]

            index_max = argmax(V_M[-1])
            index_res = argmax(EXPECTED)

            if index_max != index_res:
                errors += 1

        return errors / float(len(X))

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
#normalize_standarize(X, Y)

training_specs = TrainingSpecs(0.001, 600, 0.01, True, 0)
training_error_by_epoch = []

X = map(lambda x: np.insert(x, 0, -1), X)

#Inicializamos perceptron,
hidden_layers = [Layer(len(X[0]), 20, v_tanh, v_tanh_deriv, True)]
output_layer = Layer(21, 2, lambda x: x,  lambda x: 1, True)
ppm = Classifier(len(X[0]), hidden_layers, output_layer)

#Entrenamos y validamos.
training_error_by_epoch = ppm.train(X[:20], Y[:20], training_specs)
validation_error = ppm.out(X[:20], Y[:20]) #llama out en vez de validate porque se mide diferente el error final

# Plot de error de entrenamiento
plt.plot(range(1, len(training_error_by_epoch)+1), training_error_by_epoch, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Epoch Error')
plt.show()

# Ya veremos como printear el error de validacion.
print("VALIDATION ERROR:")
print(validation_error)
