import numpy as np
from math import tanh

import matplotlib.pyplot as plt
from Parser import parse_ej2

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

#TODO sacar?
def areConsistentLayersSpecs(pattern_size, hidden_layers_specs):
	
	if len(hidden_layers_specs) == 0:
		raise ValueError("There must be at least one Hidden Layer.")
	"""
	if pattern_size != hidden_layers_specs[0].perceptrons_input_size:
		raise ValueError("Pattern Size must be equals to First Hidden Layer Perceptron's Input Size.")

	curr = 0
	
	while curr != len(hidden_layers_specs) - 1:
		nextl = curr + 1
	
		if hidden_layers_specs[curr].perceptrons_count != hidden_layers_specs[nextl].perceptrons_input_size:
			raise ValueError("Layers specifications are inconsistent.")

		curr = nextl

class HiddenLayerSpecs(object):

	def __init__(self, perceptrons_count, perceptron_input_size):

		self.perceptrons_count = perceptrons_count
		self.perceptrons_input_size = perceptron_input_size
	"""

class PerceptronMulticapa(object):

		def __init__(self, pattern_size, hidden_layers_specs, output_size):     
	
			self.pattern_size = pattern_size

			input_size = lambda l: hidden_layers_specs[l-1] if l > 0 else pattern_size
			self.hidden_layers_Ws = [ [np.random.rand(1 + input_size(layer_ind)) for _ in range(0,layer_specs) ] for layer_ind, layer_specs in enumerate(hidden_layers_specs)]

			last_hidden_layer_specs = hidden_layers_specs[-1] 
			self.output_layer_W = [np.random.rand(1 + last_hidden_layer_specs) for _ in range(0, output_size)]
			

		def train(self, X, Y, activation_fx, activation_fx_derivative, eta=0.1, epochs=10000, epsilon=0.1):

			#agrega -1 a cada patron en x para el bias
			for input in X:
				input.append(-1)

			self.eta = eta
			self.epochs = epochs
			for i in range(self.epochs):
				for x, expected in zip(X, Y):  
					V = self.calculate_V(x, activation_fx)

					self.back_propagation(V, expected, activation_fx_derivative)

		def out(self, X, Y, activation_fx):
			error = 0
			for a, EXPECTED in zip(X, Y):
				V = self.calculate_V(a, activation_fx)
				V_M = V[-1]
				print(V_M)

			return self

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

#Los input size de los perceptrones no consideran el bias. (CANTIDAD_PERCEPTRONES, CANTIDAD_DE_ENTRADAS_SIN_BIAS_X_PPN)

#para testear localmente o con el e2 posta
if True:
	hidden_layer_specs = [2, 3]
	#Lo necesario para el XOR.

	X = [[0,0], [0,1], [1,0], [1,1]]
	Y = [[0],[1],[1],[0]]

	ppm = PerceptronMulticapa(len(X[0]), hidden_layer_specs, len(Y[0]))
	ppm.train(X, Y, binary_sigmoidal, binary_sigmoidal_derivative)
	ppm.out(X, Y, binary_sigmoidal)
else:
	X, Y = parse_ej2()
	hidden_layer_specs = [2, 2]
	ppm = PerceptronMulticapa(len(X[0]), hidden_layer_specs, len(Y[0]))
	ppm.train(X, Y, binary_sigmoidal, binary_sigmoidal_derivative)
	ppm.out(X, Y, binary_sigmoidal)