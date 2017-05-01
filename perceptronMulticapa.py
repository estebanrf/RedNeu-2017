import numpy as np
from math import tanh
from random import shuffle

import matplotlib.pyplot as plt
from Parser import parse_ej2


class TrainingSpecs(object):
	
	def __init__(self, activation_fx, activation_fx_derivative, eta, epochs, epsilon, must_train_in_training_set_order):     

		self.activation_fx = activation_fx
		self.activation_fx_derivative = activation_fx_derivative
		self.eta = eta
		self.epochs = epochs
		self.epsilon = epsilon
		self.must_train_in_training_set_order = must_train_in_training_set_order

class HiddenLayerSpecs(object):

	def __init__(self, perceptrons_count, perceptron_input_size):

		self.perceptrons_count = perceptrons_count
		self.perceptrons_input_size = perceptron_input_size

class PerceptronMulticapa(object):

		def __init__(self, pattern_size, use_uniform_distr_to_init_W, hidden_layers_specs, output_size):     
	
			are_consistent_layers_specs(pattern_size, hidden_layers_specs)

			self.pattern_size = pattern_size
			
			self.hidden_layers_Ws = [ [self.init_neuron_w(layer_specs.perceptrons_input_size, use_uniform_distr_to_init_W) for _ in range(0,layer_specs.perceptrons_count) ] for layer_specs in hidden_layers_specs]

			last_hidden_layer_specs = hidden_layers_specs[-1] 
			self.output_layer_W = [self.init_neuron_w(1 + last_hidden_layer_specs.perceptrons_count, use_uniform_distr_to_init_W) for _ in range(0, output_size)]
		
		def init_neuron_w(self, weights_size, use_uniform_distr_to_init_W):
			#No pude setear la varianza que se recomendo en la clase, no se si cambia en algo.
			if use_uniform_distr_to_init_W:
				return np.random.uniform(0.0, 1.0, weights_size)
			return np.random.rand(weights_size)

		def train(self, X, Y, training_specs):

			self.eta = training_specs.eta
			self.epochs = training_specs.epochs
			
			training_set = zip(X, Y)
			
			training_error_by_epoch = []

			for _ in range(self.epochs):
				epoch_error = 0

				if training_specs.must_train_in_training_set_order:
					shuffle(training_set)

				for x, expected in training_set:  
					V = self.calculate_V(x, training_specs.activation_fx)
					self.back_propagation(V, expected, training_specs.activation_fx_derivative)
					epoch_error += (np.linalg.norm(np.subtract(V[-1], expected)) / 2)
				
				training_error_by_epoch.append(epoch_error / len(training_set))
				#Error de una epoca = promedio del error (no se si esta bien)
				if (epoch_error / len(training_set)) <= training_specs.epsilon:
					break

			return training_error_by_epoch

		def validate(self, X, Y, activation_fx):
			error = 0 
			for x, expected in zip(X,Y):
				V = self.calculate_V(x, activation_fx)
				error += (np.linalg.norm(np.subtract(V[-1], expected)) / 2)
			return error

		def calculate_V(self, x, activation_function):
			
			V = []
			V_i = x
			i = 0
			for current_layer_W in self.hidden_layers_Ws:
				V_i_plus_1 = activation_function(np.dot(V_i, np.transpose(current_layer_W)))
				V.append(V_i)
				V_i = (np.insert(V_i_plus_1,0,-1))

			V_M = activation_function(np.dot(V_i, np.transpose(self.output_layer_W)))
			V.append(V_i)
			V.append(V_M)
			return V
		

		def back_propagation(self, V, expected, activation_fx_derivative):

			V_M = V[-1]
			V_M_minus_1 = V[-2]
			error  = np.subtract(expected,V_M)
			current_deltas = np.multiply(activation_fx_derivative(np.dot(V_M_minus_1, np.transpose(self.output_layer_W))), error)

			propagation = np.dot(np.transpose(self.output_layer_W)[1:], current_deltas) 
			self.output_layer_W += self.eta * np.outer(current_deltas, V_M_minus_1)		

			for hidden_layer_index, V_i_minus_1 in enumerate(reversed(V[:-2])):
					#Iteramos las capas ocultas en reversa tambien, esa es la razon del calculo del indice
					hidden_layer_index = (hidden_layer_index + 1) * -1
					current_deltas = np.multiply(activation_fx_derivative(np.dot(V_i_minus_1, np.transpose(self.hidden_layers_Ws[hidden_layer_index]))), propagation)
					propagation = np.dot(np.transpose(self.hidden_layers_Ws[hidden_layer_index])[1:], current_deltas)
					self.hidden_layers_Ws[hidden_layer_index] += (self.eta * np.outer(current_deltas, V_i_minus_1))

#Funciones de activacion y sus derivadas, agregar mas. ----------------------------------------------------------------
def ReLU(x):
    return np.maximum(x, 0, x)

def derivative_ReLU(x, epsilon=0.1):
    gradients = 1. * (x > 0)
    gradients[gradients == 0] = epsilon
    return gradients

def tanh_deriv(x):
    return (1 - tanh(x)**2)

def binary_sigmoidal(x):
	return 1.0 / (1 + np.exp(-1 * x))

def binary_sigmoidal_derivative(x):
	return binary_sigmoidal(x) * (1 - binary_sigmoidal(x))
#----------------------------------------------------------------------------------------------------------------------

def are_consistent_layers_specs(pattern_size, hidden_layers_specs):
	
	if len(hidden_layers_specs) == 0:
		raise ValueError("There must be at least one Hidden Layer.")

	if pattern_size != hidden_layers_specs[0].perceptrons_input_size:
		raise ValueError("Pattern Size must be equals to First Hidden Layer Perceptron's Input Size.")

	curr = 0
	
	while curr != len(hidden_layers_specs) - 1:
		nextl = curr + 1
	
		if hidden_layers_specs[curr].perceptrons_count + 1 != hidden_layers_specs[nextl].perceptrons_input_size:
			raise ValueError("Layers specifications are inconsistent.")

		curr = nextl

#----------------------------------------------------------------------------------------------------------------------

training_specs = TrainingSpecs(binary_sigmoidal, binary_sigmoidal_derivative, 0.1, 10, 0.25, True)
training_error_by_epoch = []
validation_error = -1
X = []
Y = []

#para testear localmente XOR o con el ejercicio 2 
if False:
	#Lo necesario para el XOR.
	X = [(-1,0,0), (-1,0,1), (-1,1,0) , (-1, 1,1)]
	Y = [[0],[1],[1],[0]]
else:
	X, Y = parse_ej2()
	X = map(lambda x: np.insert(x, 0, -1), X)



#Inicializamos perceptron, entrenamos y validamos.
hidden_layer_specs = [HiddenLayerSpecs(4, len(X[0]))]
ppm = PerceptronMulticapa(len(X[0]), True, hidden_layer_specs, len(Y[0]))
training_error_by_epoch = ppm.train(X, Y, training_specs)
validation_error = ppm.validate(X, Y, training_specs.activation_fx)

#Plot de error de entrenamiento
plt.plot(range(1, len(training_error_by_epoch)+1), training_error_by_epoch, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Epoch Error')
plt.show()

#Ya veremos como printear el error de validacion.
print("VALIDATION ERROR:")
print(validation_error / len(X))
