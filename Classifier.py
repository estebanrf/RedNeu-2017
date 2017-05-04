from perceptronMulticapa import PerceptronMulticapa, Layer, TrainingSpecs, v_tanh, v_tanh_deriv
import matplotlib.pyplot as plt
from numpy import argmax
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
