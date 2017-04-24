# Redes Neuronales, DC, UBA - Primer Cuatrimestre 2017
# Codigo template simplificado de perceptron simple escalon que resuelve el AND logico.
# MUY IMPORTANTE: Modificar el codigo para incorporarle las mejoras especificadas en los comentarios del codigo.

import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.01

class PerceptronSimple(object):


    def __init__(self, input_size, output_size):     

        self.w_ = np.random.rand(1 + input_size, output_size)

    def train(self, X, Y, activation_fx, activation_fx_derivative, eta=0.1, epochs=100):
        
        self.eta = eta
        self.epochs = epochs
        # MEJORAS SOLICITADAS: Agregar que tambien se guarde el historial de los errores del conjunto de validacion.
        self.epoch_errors = []

        for _ in range(self.epochs):
            current_epoch_error = 0 
        
            # MEJORAS SOLICITADAS: Modificarlo para que el orden de los patrones de entrenamiento sea random en cada epoca 
            # (estrategia simple que mejora la velocidad de aprendizaje de la red).
            for x, expected in zip(X, Y):  
                          
                error = expected - self.predict(x,activation_fx)
                
                norm_error = np.linalg.norm(error)
                current_epoch_error += norm_error

                update = self.eta * np.multiply(error, activation_fx_derivative(self.net_input(x)))
               
                delta = np.transpose(np.outer(update, x))
               
              	self.w_[1:] += delta
                
                # En realidad es Update por np.ones (delta correspondiente al bias)
                self.w_[0] +=  update 
   
            self.epoch_errors.append(current_epoch_error)

            if current_epoch_error <= EPSILON:
                break
            
        return self

    def net_input(self, x):

        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x, activation_function):
        
        return activation_function(self.net_input(x))


def heaviside_step(x):
    return np.where(x >= 0.0, 1, 0)

def heaviside_step_derivative(x):
	return 1

def binary_sigmoidal(x):
	return 1.0 / (1 + np.exp(-1 * x))

def binary_sigmoidal_derivative(x):
	return binary_sigmoidal(x) / (1 - binary_sigmoidal(x))

def bipolar_sigmoidal(x):
    return (2.0 / (1 + np.exp(-1 * x))) - 1
#Falta la derivada de la bipolar simoidal


X = np.array([(0,1), (0,0), (1, 0), (1, 1)])
# Usa salidas deseadas unipolares. 
# MEJORAS SOLICITADAS: Probar con salidas bipolares, y ver si mejora la performance de clasificacion 
# (ojo que si la salida deseada es bipolar, la funcion de activacion de las unidades de salida tambien debe ser bipolar)
Y = np.array([(1,0), (0,0), (1,0), (1,1)])

training_input_size = X.shape[1]
training_output_size = Y.shape[1]

ppn = PerceptronSimple(training_input_size, training_output_size)

# MEJORAS SOLICITADAS: Experimentar con otra cantidad de epocas y factor de aprendizaje.
ppn.train(X, Y, binary_sigmoidal, binary_sigmoidal_derivative, epochs=100, eta=10 ** -1)

# Se grafica la cantidad de clasificaciones incorrectas en cada epoca versus numero de epoca. 
# MEJORAS SOLICITADAS:
# Mejora 2) Agregar al grafico el error de costo del conjunto de validacion (o sea, debe graficar en el mismo grafico, ambos errores de costo)
plt.plot(range(1, len(ppn.epoch_errors)+1), ppn.epoch_errors, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Epoch Error')
plt.show()

# MEJORAS SOLICITADAS: Agregar imprimir la funcion de costo y el root-mean-square error final del conjunto de testing, 
# para saber la performance final de la red neuronal.