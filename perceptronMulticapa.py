import numpy as np
from math import tanh
from random import shuffle

import matplotlib.pyplot as plt
from Parser import parse_ej1, parse_ej2, normalize_standarize, normalize_minmax

class TrainingSpecs(object):

	def __init__(self, eta, epochs, epsilon, must_train_in_training_set_order, momentum_inertia, subsets_quantity_for_minibatch, adaptative_params):

		self.eta = eta
		self.epochs = epochs
		self.epsilon = epsilon
		self.must_train_in_training_set_order = must_train_in_training_set_order
		
		if momentum_inertia < 0 or momentum_inertia > 1:
			raise ValueError("Momentum inertia must be in [0,1]")
		self.momentum_inertia = momentum_inertia
 		
 		self.subsets_quantity_for_minibatch = subsets_quantity_for_minibatch
 		self.adaptative_params = adaptative_params

class AdaptativeParameters(object):

	def __init__(self, count_of_errors_straight, a, beta):

		self.count_of_errors_straight = count_of_errors_straight
		self.a = a
		self.beta = beta	

class Layer(object):

	def __init__(self, layer_input_size, neurons_count, activation_fx,
				 activation_fx_derivative, use_uniform_distr_to_init_W):

		self.activation_fx = activation_fx
		self.activation_fx_derivative = activation_fx_derivative

		if use_uniform_distr_to_init_W:
			self.neurons_matrix = np.random.uniform(0.0, 1.0, (layer_input_size, neurons_count))
		else:
			self.neurons_matrix = np.random.rand((layer_input_size, neurons_count))

		self.previous_momentum_delta_W = np.zeros((layer_input_size, neurons_count))

class PerceptronMulticapa(object):

		def __init__(self, hidden_layers, output_layer):

			self.hidden_layers = hidden_layers
			self.output_layer = output_layer

		def train(self, X_training, Y_training, X_validation, Y_validation, training_specs):

			self.eta = training_specs.eta
			self.epochs = training_specs.epochs
			self.momentum_inertia = training_specs.momentum_inertia
			self.adaptative_params = training_specs.adaptative_params

			training_set = zip(X_training, Y_training)
			validation_set = zip(X_validation, Y_validation)

			training_set = self.prepare_batch_or_mini_batch_training_set(training_set, training_specs.subsets_quantity_for_minibatch)
			
			training_error_by_epoch = []
			validation_error_by_epoch = []

			training_subset_index = 0

			error_reference_from_past = -1
			self.error_difference_counting = 0

			for _ in range(self.epochs):
				
				current_training_set = training_set[training_subset_index]
				
				if not training_specs.must_train_in_training_set_order:
					shuffle(current_training_set)
				
				for x, expected in current_training_set:
					V = self.calculate_V(x)
					self.back_propagation(V, expected)

				epoch_training_error = self.calculate_error_using(current_training_set)
				epoch_validation_error = self.calculate_error_using(validation_set)

				training_error_by_epoch.append(epoch_training_error)
				validation_error_by_epoch.append(epoch_validation_error)

				if self.adaptative_params.count_of_errors_straight > 0:
					new_error_reference = self.adapt_eta(error_reference_from_past, epoch_training_error)
					if new_error_reference > 0:
						error_reference_from_past = new_error_reference
						self.error_difference_counting = 0


				#Si el error de validacion de la epoca es menor que un EPSILON terminamos.
				if epoch_validation_error <= training_specs.epsilon:
					break

				training_subset_index = (training_subset_index + 1) % len(training_set)

			return (training_error_by_epoch,validation_error_by_epoch)
		
		def adapt_eta(self, error_reference_from_past, epoch_training_error):

			if error_reference_from_past == -1:
					#Es el primer entrenamiento.
					return epoch_training_error
			else:
				if error_reference_from_past < epoch_training_error:
					#Hubo mas error
					if self.error_difference_counting < 0:
						#Venia una seguidilla de menos error, y ahora reseteamos.
						return epoch_training_error
					
					self.error_difference_counting = self.error_difference_counting + 1

					if self.adaptative_params.count_of_errors_straight == self.error_difference_counting:
						self.eta += -1 * self.eta * self.adaptative_params.beta
						return epoch_training_error
					else:
						return 0
				else:
					self.error_difference_counting = self.error_difference_counting - 1
					#Hubo menos error	
					if self.error_difference_counting > 0:
						return epoch_training_error
				
					self.error_difference_counting = self.error_difference_counting - 1
					
					if self.adaptative_params.count_of_errors_straight == abs(self.error_difference_counting):
						self.eta += self.adaptative_params.a
						return epoch_training_error
					else:
						return 0


		def prepare_batch_or_mini_batch_training_set(self, training_set, subsets_quantity_for_minibatch):

			if subsets_quantity_for_minibatch == 0 or subsets_quantity_for_minibatch == 1:
				#Se corre todo el set de entrenamiento por epoca, o sea habra un unico subset.
				training_set = [training_set]
			else:
				if len(training_set) < subsets_quantity_for_minibatch:
					raise ValueError("There are not enough training patterns to create the specified subsets.")
				#Dividimos el set de entrenamiento en la cantidad de subsets especificadas para minibatch.
				elements_per_subset = len(training_set) / subsets_quantity_for_minibatch
				training_set = [training_set[i : i + elements_per_subset] for i in range(0, len(training_set), elements_per_subset)]
			return training_set

		def calculate_error_using(self, zip_Xs_Ys):
			error = 0
			for x, expected in zip_Xs_Ys:
				V = self.calculate_V(x)
				error += (np.linalg.norm(np.subtract(V[-1], expected)) / 2)

			return error / float(len(zip_Xs_Ys))

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

			delta_W =  self.eta * np.outer(V_M_minus_1, current_delta) - (self.momentum_inertia * self.output_layer.previous_momentum_delta_W)
			self.output_layer.neurons_matrix += delta_W
			self.output_layer.previous_momentum_delta_W = delta_W
			V_index = -2

			for hidden_layer in reversed(self.hidden_layers):
				#Iteramos las capas ocultas en reversa tambien, esa es la razon del calculo del indice
				# como ya propagamos por la capa de salida, ahora empezamos desde el -2 en V
				V_i_minus_1 = V[V_index - 1]
				current_delta = np.multiply(hidden_layer.activation_fx_derivative(np.dot(V_i_minus_1, hidden_layer.neurons_matrix)), propagation)
				propagation = np.dot(current_delta, np.transpose(hidden_layer.neurons_matrix[1:]))
				delta_W = self.eta * np.outer(V_i_minus_1, current_delta) - (self.momentum_inertia * hidden_layer.previous_momentum_delta_W)
				hidden_layer.neurons_matrix += delta_W
				hidden_layer.previous_momentum_delta_W = delta_W
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
# eta, epochs, epsilon, must_train_in_training_set_order, momentum_inertia, subsets_quantity_for_minibatch, adaptative_params (ES UNA TRIPLA)
#### Para deshabilitar parametros adaptativos poner como primer parametro del constructor un -1.
training_specs = TrainingSpecs(0.1, 100, 0.00001, True, 0, 1, AdaptativeParameters(3, 0.001, 0.1))

#para testear localmente paridad o con el ejercicio 2
EJERCICIO = 1
if EJERCICIO == 0:
	#Lo necesario para el XOR.
	X_tr = X_valid = [[-1,0,0,0], [-1,0,1,0], [-1,1,0,0] , [-1, 1,1,0],
		 [-1,0,0,1], [-1,0,1,1], [-1,1,0,1] , [-1, 1,1,1]]
	Y_tr = Y_valid = [[0],[1],[1],[0],[1],[0],[0],[1]]
elif EJERCICIO == 1:
	X_tr, Y_tr, X_valid, Y_valid, X_test, Y_test = parse_ej1(percent_train=80, percent_valid=10,
                                                             f_normalize_X=normalize_standarize, f_normalize_Y=None)
else:
    X_tr, Y_tr, X_valid, Y_valid, X_test, Y_test = parse_ej2(percent_train=80, percent_valid=10,
                                                             f_normalize_X =None, f_normalize_Y = None)

#Inicializamos perceptron,
hidden_layers = [Layer(len(X_tr[0]), 10, binary_sigmoidal, binary_sigmoidal_derivative, True)]
output_layer = Layer(11, 2, binary_sigmoidal, binary_sigmoidal_derivative, True)
ppm = PerceptronMulticapa(hidden_layers, output_layer)

#Entrenamos y validamos.
error_by_epoch = ppm.train(X_tr, Y_tr, X_valid, Y_valid, training_specs)

# Plot de error de entrenamiento
plt.plot(range(1, len(error_by_epoch[0])+1), error_by_epoch[0], marker='o', label="Training error")
plt.plot(range(1, len(error_by_epoch[1])+1), error_by_epoch[1], marker='o', label="Validation error")
plt.legend(loc='upper left')
plt.annotate(error_by_epoch[0][-1], xy = (len(error_by_epoch[0]) + 1, error_by_epoch[0][-1]), bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
plt.annotate(error_by_epoch[1][-1], xy = (len(error_by_epoch[0]) + 1, error_by_epoch[1][-1]), bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

plt.xlabel('Epoch')
plt.ylabel('Epoch Error')

plt.show()
