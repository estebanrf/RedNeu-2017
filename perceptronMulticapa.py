import numpy as np
from math import tanh
from random import shuffle

import matplotlib.pyplot as plt
from Parser import parse_ej2, normalize_standarize, normalize_minmax

class TrainingSpecs(object):

	def __init__(self, eta, epochs, epsilon, must_train_in_training_set_order, momentum_inertia):

		self.eta = eta
		self.epochs = epochs
		self.epsilon = epsilon
		self.must_train_in_training_set_order = must_train_in_training_set_order
		if momentum_inertia < 0 or momentum_inertia > 1:
			raise ValueError("Momentum inertia must be in [0,1]")
 		self.momentum_inertia = momentum_inertia

class Layer(object):

	def __init__(self, layer_input_size, neurons_count, activation_fx,
				 activation_fx_derivative, use_uniform_distr_to_init_W):

		self.layer_input_size = layer_input_size
		self.neurons_count = neurons_count
		self.activation_fx = activation_fx
		self.activation_fx_derivative = activation_fx_derivative
		if use_uniform_distr_to_init_W:
			self.neurons_matrix = np.random.uniform(0.0, 1.0, (layer_input_size, neurons_count))
		else:
			self.neurons_matrix = np.random.rand((layer_input_size, neurons_count))
		self.previous_momentum_neurons_matrix = np.zeros((layer_input_size, neurons_count))

class PerceptronMulticapa(object):

		def __init__(self, pattern_size, hidden_layers, output_layer):

			# no creo que sea necesario lo del pattern_size, me parece que
			# estaria mejor hacer un pre chequeo de nuestro set de entrenamiento
			# y si vemos que no coincide con la primer capa oculta ahi si que chotee.

			# are_consistent_layers_specs(pattern_size, hidden_layers_specs)

			self.pattern_size = pattern_size
			self.hidden_layers = hidden_layers
			self.output_layer = output_layer

		def train(self, X, Y, training_specs):

			self.eta = training_specs.eta
			self.epochs = training_specs.epochs
			self.momentum_inertia = training_specs.momentum_inertia
			training_set = zip(X, Y)

			training_error_by_epoch = []
			for _ in range(self.epochs):
				epoch_error = 0

				if not training_specs.must_train_in_training_set_order:
					shuffle(training_set)

				for x, expected in training_set:
					V = self.calculate_V(x)
					self.back_propagation(V, expected)
					# epoch_error += (np.linalg.norm(np.subtract(V[-1], expected)) / 2)

				epoch_error = self.validate(X,Y)

				training_error_by_epoch.append(epoch_error)
				#Error de una epoca = promedio del error (no se si esta bien)
				if (epoch_error / float(len(training_set))) <= training_specs.epsilon:
					break

			return training_error_by_epoch

		def validate(self, X, Y):
			error = 0
			for x, expected in zip(X,Y):
				V = self.calculate_V(x)
				error += (np.linalg.norm(np.subtract(V[-1], expected)) / 2)

			return error / float(len(X))

		def calculate_V(self, x):

			V = []
			V_i = x
			# en V guardo el patron de entrenamiento y las salidas de las capas
			for current_layer in self.hidden_layers:
				V.append(V_i)
				V_i = current_layer.activation_fx(np.dot(V_i, current_layer.neurons_matrix))
				V_i = np.insert(V_i,0,-1)

			V.append(V_i)
			V_M = self.output_layer.activation_fx(np.dot(V_i, self.output_layer.neurons_matrix))
			V.append(V_M)
			return V

		def back_propagation(self, V, expected):

			V_M = V[-1]
			V_M_minus_1 = V[-2]
			error  = np.subtract(expected,V_M)
			current_delta = np.multiply(self.output_layer.activation_fx_derivative(np.dot(V_M_minus_1, self.output_layer.neurons_matrix)), error)
			propagation = np.dot(current_delta, np.transpose(self.output_layer.neurons_matrix[1:]))

			delta_W = (1 if self.momentum_inertia == 0 else -1) * self.eta * np.outer(V_M_minus_1, current_delta) - (self.momentum_inertia * self.output_layer.previous_momentum_neurons_matrix)
			self.output_layer.neurons_matrix += delta_W
			self.output_layer.previous_momentum_neurons_matrix = delta_W
			V_index = -2

			for hidden_layer in reversed(self.hidden_layers):
				#Iteramos las capas ocultas en reversa tambien, esa es la razon del calculo del indice
				# como ya propagamos por la capa de salida, ahora empezamos desde el -2 en V
				V_i_minus_1 = V[V_index - 1]
				current_delta = np.multiply(hidden_layer.activation_fx_derivative(np.dot(V_i_minus_1, hidden_layer.neurons_matrix)), propagation)
				propagation = np.dot(current_delta, np.transpose(hidden_layer.neurons_matrix[1:]))
				delta_W = (1 if self.momentum_inertia == 0 else -1) * self.eta * np.outer(V_i_minus_1, current_delta) - (self.momentum_inertia * hidden_layer.previous_momentum_neurons_matrix)
				hidden_layer.neurons_matrix += delta_W
				hidden_layer.previous_momentum_neurons_matrix = delta_W
				V_index = V_index - 1

#Funciones de activacion y sus derivadas, agregar mas. ----------------------------------------------------------------
def ReLU(x):
    return np.maximum(x, 0, x)

def derivative_ReLU(x, epsilon=0.1):
    gradients = 1. * (x > 0)
    gradients[gradients == 0] = epsilon
    return gradients

def v_tanh(x): #redefinida para soportar vectores
	return map(tanh, x)

def v_tanh_deriv(x):
    return map(lambda v: 1 - tanh(v)**2, x)

def binary_sigmoidal(x):
	x = np.array(x)
	return 1.0 / (1 + np.exp(-2 * x))

def binary_sigmoidal_derivative(x):
	x = np.array(x)
	return 2 * binary_sigmoidal(x) * (1 - binary_sigmoidal(x))
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
# eta, epochs, epsilon, must_train_in_training_set_order, momentum_inertia
training_specs = TrainingSpecs(0.1, 1000, 0.00001, True, 0)
training_error_by_epoch = []
validation_error = -1

#para testear localmente paridad o con el ejercicio 2
if True:
	#Lo necesario para el XOR.
	X_tr = X_valid = [[-1,0,0,0], [-1,0,1,0], [-1,1,0,0] , [-1, 1,1,0],
		 [-1,0,0,1], [-1,0,1,1], [-1,1,0,1] , [-1, 1,1,1]]
	Y_tr = Y_valid = [[0],[1],[1],[0],[1],[0],[0],[1]]
else:

	def norm(X,Y):
		vector_divisor_X = np.array([1, 1000, 1000, 1000, 10, 10, 1, 10])
		vector_divisor_Y = np.array([100, 100])
		X = X / vector_divisor_X
		Y = Y / vector_divisor_Y
		return X, Y

	X_tr, Y_tr, X_valid, Y_valid, X_test, Y_test = parse_ej2(percent_train=70, percent_valid=10, f_normalize=None)


#Inicializamos perceptron,
hidden_layers = [Layer(len(X_tr[0]), 10, binary_sigmoidal, binary_sigmoidal_derivative, True)]
# output_layer = Layer(11, 2, lambda x: x,  lambda x: 1, True)
output_layer = Layer(11, 2, binary_sigmoidal, binary_sigmoidal_derivative, True)
ppm = PerceptronMulticapa(len(X_tr[0]), hidden_layers, output_layer)

#Entrenamos y validamos.
training_error_by_epoch = ppm.train(X_tr, Y_tr, training_specs)
validation_error = ppm.validate(X_valid, Y_valid)

# Plot de error de entrenamiento
plt.plot(range(1, len(training_error_by_epoch)+1), training_error_by_epoch, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Epoch Error')
plt.show()

# Ya veremos como printear el error de validacion.
print("VALIDATION ERROR:")
print(validation_error)
