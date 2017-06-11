# Redes Neuronales, DC, UBA - Primer Cuatrimestre 2017
# Codigo template simplificado de perceptron simple escalon que resuelve el AND logico.
# MUY IMPORTANTE: Modificar el codigo para incorporarle las mejoras especificadas en los comentarios del codigo.

import numpy as np
import matplotlib.pyplot as plt

from Parser import normalize_standarize, normalize_minmax

EPSILON = 0.01

class PerceptronSimple(object):


    def __init__(self, input_size, output_size):     

        self.w_ = np.random.rand(input_size, output_size)
        self.output_size = output_size

    def train(self, X, activation_fx, activation_fx_derivative, t=1000):
        
        while t > 0:
            eta = 1/t
            for x in X:  
                y = self.predict(x,activation_fx)
                x_aprox = calculate_x_aproximation_with_sanger(y)               
                delta = eta * y * (x - x_aprox) 
                self.w_ += delta          	
                t = t - 1
        return self

    def net_input(self, x):

        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x, activation_function):
        
        return activation_function(self.net_input(x))

    def calculate_x_aproximation_with_sanger(self, y):
        x_aprox = []

        for j in range(1, self.output_size + 1)
            x_aprox.append(np.multiply(y[0:j], self._w[0:j]).sum(axis=1))

        return x_aprox

X = np.array([(0,1), (0,0), (1, 0), (1, 1)])


training_input_size = X.shape[1]
training_output_size = Y.shape[1]

ppn = PerceptronSimple(training_input_size, training_output_size)

ppn.train(X, , t=1000)

plt.plot(range(1, len(ppn.epoch_errors)+1), ppn.epoch_errors, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Epoch Error')
plt.show()
