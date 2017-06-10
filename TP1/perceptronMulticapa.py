#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from math import tanh
from random import shuffle
from ast import literal_eval
import os, json

import matplotlib.pyplot as plt
from Parser import parse_ej1, parse_ej2, normalize_standarize, normalize_minmax

# cargamos los datos con los cuales se va a entrenar nuestro perceptron
def parse_layers(args, IN_SIZE, OUT_SIZE):

	args.l_hidden = literal_eval(args.l_hidden) #pasa a lista posta
	args.l_hidden = [Layer(1+args.l_hidden[ind-1].neurons_count if ind > 0 else IN_SIZE, a_layer[0],
					binary_sigmoidal if a_layer[1] == 1 else v_tanh,
					binary_sigmoidal_derivative if a_layer[1] == 1 else v_tanh_deriv,
					True)
					for ind, a_layer in enumerate(args.l_hidden)]


	args.l_output = literal_eval(args.l_output)
	args.l_output = Layer(1+args.l_hidden[-1].neurons_count, OUT_SIZE,
					binary_sigmoidal if args.l_output == 1 else v_tanh,
					binary_sigmoidal_derivative if args.l_output == 1 else v_tanh_deriv,
					True)

def parse_arguments():
	arguments = argparse.ArgumentParser()
	arguments.add_argument('--eta', type=float, help="eta inicial") #defaultea a None si no se le define
	arguments.add_argument('--epochs', type=int, help="cantidad de epocas")
	arguments.add_argument('--shuffle', type=int, help="randomizar orden de patrones (0 o 1)")
	arguments.add_argument('--momentum', type=float, help="momentum inertia")
	arguments.add_argument('--batch_size', type=int, help="tamaño del batch (para batch, -1)")
	arguments.add_argument('--adapt', type=int, help="usar parametros adaptativos (0 o 1)")
	arguments.add_argument('--ej', type=int, help="numero de ejercicio")
	arguments.add_argument('--norm_x', type=int, help="normalizar input (0 o 1)")
	arguments.add_argument('--norm_y', type=int, help="normalizar output (0 o 1)")
	arguments.add_argument('--l_hidden',
						help="""
						hidden layer. ejemplo: [layer1,layer,...] con layerX = (cantNeuronas,activacion),
						con activation=1 para sigmoidea estandar, =2 para tanh
						""")
	arguments.add_argument('--l_output', help="1 o 2 para activacion sigmoidea o tanh")
	arguments.add_argument('--list', type=int, help="Para mostrar las redes entrenadas (0 o 1)")
	arguments.add_argument('--test', help="Red entrenada para testear (mandar --ej tambien)")
	arguments.add_argument('--export', help="Nombre, si se desea exportar la red")
	args = arguments.parse_args()

	process = lambda b, r1, r2: r1 if b == 1 else r2
	if args.shuffle is not None: args.shuffle = process(args.shuffle, True, False)
	if args.adapt is not None: args.adapt = process(args.adapt, True, False)
	if args.list is not None: args.list = process(args.list, True, False)
	if args.norm_x is not None: args.norm_x = process(args.norm_x, normalize_standarize, None)
	if args.norm_y is not None: args.norm_y = process(args.norm_x, normalize_standarize, None)

	return args

def network_import(directory):

	filenames = sorted([fn for fn in os.listdir(directory) if '.npy' not in fn])
	layers = []
	for filename in filenames:

		prop = json.load(open(directory+'/'+filename, 'r'))

		fx_name = prop["activation_fx"]
		fx = globals()[fx_name]

		matrix_filename = prop["neurons_matrix_file"]
		matrix = np.load(matrix_filename+'.npy')

		layers.append(ExistingLayer(fx, matrix))

	return PerceptronMulticapa(layers[:-1], layers[-1])

def network_export(net, network_name):

	network_directory = 'net-' + network_name + '/'

	if not os.path.exists(network_directory):
	    os.makedirs(network_directory)

	for ind, l in enumerate(net.hidden_layers + [net.output_layer]):

		matrix_filename = network_directory + str(ind)

		np.save(matrix_filename, l.neurons_matrix)
		prop = {"activation_fx": l.activation_fx.__name__,
				"neurons_matrix_file": matrix_filename}
		outstring = json.dumps(prop)

		with open(network_directory+str(ind)+'.json', 'w') as f:
			f.write(outstring)

def list_trained_networks():
	print "Redes existentes:"
	print filter(lambda x: 'net-' in x, next(os.walk('.'))[1])

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

# crea una capa con toda su arquitectura
class Layer(object):

	def __init__(self, layer_input_size, neurons_count, activation_fx,
				 activation_fx_derivative, use_uniform_distr_to_init_W):

		self.activation_fx = activation_fx
		self.activation_fx_derivative = activation_fx_derivative
		self.neurons_count = neurons_count

		if use_uniform_distr_to_init_W:
			self.neurons_matrix = np.random.uniform(0.0, 1.0, (layer_input_size, neurons_count))
		else:
			self.neurons_matrix = np.random.rand((layer_input_size, neurons_count))

		self.previous_momentum_delta_W = np.zeros((layer_input_size, neurons_count))

class ExistingLayer(Layer):
	def __init__(self, activation_fx, neurons_matrix):
		self.activation_fx = activation_fx
		self.neurons_matrix = neurons_matrix

class PerceptronMulticapa(object):

		def __init__(self, hidden_layers, output_layer):

			self.hidden_layers = hidden_layers
			self.output_layer = output_layer

		# patrones de entrenamiento, resultados esperados, patrones para la validacion, resultados esperados, datos del entrenamiento
		def train(self, X_training, Y_training, X_validation, Y_validation, training_specs):
			self.eta = training_specs.eta
			self.epochs = training_specs.epochs
			self.momentum_inertia = training_specs.momentum_inertia
			self.adaptative_params = training_specs.adaptative_params

			training_set = zip(X_training, Y_training)
			validation_set = zip(X_validation, Y_validation)

			training_error_by_epoch = []
			validation_error_by_epoch = []

			error_reference_from_past = -1
			previous_validation_error = -1
			consecutive_increases = 0
			check_in = 1
			self.error_difference_counting = 0
			for _ in range(self.epochs):
				if not training_specs.must_train_in_training_set_order:
					shuffle(training_set)

				# divido al set de entrenamiento segun la cantidad de subconjuntos indicados
				training_set_divided = self.prepare_batch_or_mini_batch_training_set(training_set, training_specs.subsets_quantity_for_minibatch)
				# para cada subconjunto de entrenamiento voy a tener que hacer feedforward
				# y al final aplicar back_propagation
				for current_training_set in training_set_divided:
					results = []
					# hago feedforward para cada subconjunto de nuestro set de entrenamiento
					for x, expected in current_training_set:
						V = self.calculate_V(x)
						# guardo los resultados obtenidos
						results.append(V[-1])

					# ya recorri todo el subconjunto, aplico back_propagation
					self.back_propagation(V, current_training_set, results)

				# ya recorri todo el set, calculo tanto el error de entrenamiento como el de validacion
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
				#early stopping, si el error de validacion aumenta 3 veces seguidas,
				# tomando muestras cada 5 epocas, terminamos.
				if check_in == 0:
					if epoch_validation_error > previous_validation_error:
						consecutive_increases += 1
						if consecutive_increases == 3:
							break
					previous_validation_error = epoch_validation_error
					consecutive_increases = 0
					check_in = 5
				else:
					check_in -= 1

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
				# Se corre todo el set de entrenamiento por epoca, o sea habra un unico subset.
				# Esto se corresponde con hacer batch
				training_set = [training_set]
			else:
				if len(training_set) < subsets_quantity_for_minibatch:
					raise ValueError("There are not enough training patterns to create the specified subsets.")
				#Dividimos el set de entrenamiento en la cantidad de subsets especificadas para minibatch.
				if subsets_quantity_for_minibatch < 0:
					# Esta opcion es para usar estocastico
					subsets_quantity_for_minibatch = len(training_set)
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

		def back_propagation(self, V, training, results):

			# V_M va a ser el vector resultado obtenido (capa de salida)
			# V_M_minus_1 es el resultado inmediato anterior (ultima capa oculta)
			V_M = V[-1]
			V_M_minus_1 = V[-2]

			# como podemos estar corriendo con batch o minibaych, calculamos el error
			# cuadratico medio y ese error obtenido lo consideramos para calcular los deltas
			# notar que para estocastico no calculamos el error con ecm
			error = 0
			for pattern, result in zip(training, results):
				error += np.subtract(pattern[1], result)
			error = error/len(training)
			# calculamos el delta como la multiplicacion (uno a uno) entre aplicarle la derivada de
			# la funcion de activacion al resultado obtenido de la matriz de neuronas
			# de la capa de salida y el error antes calculado
			current_delta = np.multiply(self.output_layer.activation_fx_derivative(np.dot(V_M_minus_1, self.output_layer.neurons_matrix)), error)
			# calculamos la propagacion como la multiplicacion entre el current_delta y la matriz
			# traspuesta de nuestra capa de salida. Notar que tomamos la matriz de neuronas sin su bias
			propagation = np.dot(current_delta, np.transpose(self.output_layer.neurons_matrix[1:]))
			# calculamos la matriz con la cual vamos a actualizar la matriz de neuronas como
			# la constante de aprendizaje por la matriz obtenida de multiplicar el resultado
			# de la ultima capa oculta por el delta. De haber pasado como parametro un momentum
			# mayor a 0 ademas se le restara la matriz obtenida de multiplicar el momentum por
			# la matriz utilizada en la epoca anterior para actualizar la matriz
			delta_W =  self.eta * np.outer(V_M_minus_1, current_delta) - (self.momentum_inertia * self.output_layer.previous_momentum_delta_W)
			# actualizamos nuestra matriz de neuronas
			self.output_layer.neurons_matrix += delta_W
			self.output_layer.previous_momentum_delta_W = delta_W
			V_index = -2

			# continuamos con el back_propagation por el resto de las capas ocultas
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
	return np.multiply(1.7159, map(tanh, x * (2.0/3.0)))

def v_tanh_deriv(x):
    return np.multiply(1.7159, map(lambda v: 1 - tanh((2.0/3.0)*v)**2, (2.0/3.0)*x))

def binary_sigmoidal(x):
	x = np.array(x)
	return 1.0 / (1 + np.exp(-2 * x))

def binary_sigmoidal_derivative(x):
	x = np.array(x)
	return 2 * binary_sigmoidal(x) * (1 - binary_sigmoidal(x))

#----------------------------------------------------------------------------------------------------------------------
print("Correr con -h para ver opciones")
args = parse_arguments()
#parametros configurables por consola   #tocar esta columna
EJERCICIO = args.ej or 					1
ETA = args.eta or 						0.1
EPOCHS = args.epochs or 				100
TRAIN_IN_ORDER = 						True 					if args.shuffle is None else args.shuffle
MOMENTUM = args.momentum or				0
MINIABATCH_SIZE = args.batch_size or 	-1 #-1 para estocastico, 1 para batch, n para mini batch
ADPT_STRAIGHT_ERROR_COUNT = 			3 						if args.adapt is None else 0
NORM_X = 								normalize_standarize 	if args.norm_x is None else args.norm_x
NORM_Y = 								None 					if args.norm_y is None else args.norm_y
LIST_EXISTING =							False					if args.list is None else args.list
TEST_EXISTING = args.test or			None
EXPORT = args.export or			        None
#parametros no configurables por consola
ADPT_A = 								0.001
ADPT_BETA = 							0.1
EPSILON = 								0.00001


# eta, epochs, epsilon, must_train_in_training_set_order, momentum_inertia, subsets_quantity_for_minibatch, adaptative_params (ES UNA TRIPLA)
# -1 si queremos correr estocastico
# 1 si queremos corres batch
# la cantidad de subconjuntos si queremos correr minibatch
#### Para deshabilitar parametros adaptativos poner como primer parametro del constructor un -1.

training_specs = TrainingSpecs(ETA, EPOCHS, EPSILON, TRAIN_IN_ORDER, MOMENTUM, MINIABATCH_SIZE,
							AdaptativeParameters(ADPT_STRAIGHT_ERROR_COUNT, ADPT_A, ADPT_BETA))

#para testear localmente paridad o con el ejercicio 2
if EJERCICIO == 0:
	#Lo necesario para el XOR.
	X_tr = X_valid = [[-1,0,0,0], [-1,0,1,0], [-1,1,0,0] , [-1, 1,1,0],
		 [-1,0,0,1], [-1,0,1,1], [-1,1,0,1] , [-1, 1,1,1]]
	Y_tr = Y_valid = [[0],[1],[1],[0],[1],[0],[0],[1]]
elif EJERCICIO == 1:
	X_tr, Y_tr, X_valid, Y_valid, X_test, Y_test = parse_ej1(percent_train=80, percent_valid=10,
                                                             f_normalize_X=NORM_X, f_normalize_Y=NORM_Y)
else:
    X_tr, Y_tr, X_valid, Y_valid, X_test, Y_test = parse_ej2(percent_train=80, percent_valid=10,
                                                             f_normalize_X=normalize_minmax, f_normalize_Y=normalize_minmax)

training_specs = TrainingSpecs(ETA, EPOCHS, EPSILON, TRAIN_IN_ORDER,
                               MOMENTUM, MINIABATCH_SIZE,
                               AdaptativeParameters(ADPT_STRAIGHT_ERROR_COUNT, ADPT_A, ADPT_BETA))
#Inicializamos perceptron,
if not args.l_hidden or not args.l_output:
	hidden_layers = [Layer(len(X_tr[0]), 10, binary_sigmoidal, binary_sigmoidal_derivative, True)]
	output_layer = Layer(1+hidden_layers[-1].neurons_count, 2, lambda x: x, lambda x: 1, True)
else:
	parse_layers(args, len(X_tr[0]), len(Y_tr[0]))
	hidden_layers = args.l_hidden
	output_layer = args.l_output

if LIST_EXISTING:
	list_trained_networks()
elif TEST_EXISTING:
	ppm = network_import(TEST_EXISTING)
	err_tr  = ppm.calculate_error_using(zip(X_tr, Y_tr))
	err_val = ppm.calculate_error_using(zip(X_valid, Y_valid))
	err_tst = ppm.calculate_error_using(zip(X_test, Y_test))
	print "Training Error: %s" % err_tr
	print "Validation Error: %s" % err_val
	print "Testing Error: %s" % err_tst
else:
	ppm = PerceptronMulticapa(hidden_layers, output_layer)

	#Entrenamos y validamos.
	error_by_epoch = ppm.train(X_tr, Y_tr, X_valid, Y_valid, training_specs)
	if EXPORT is not None:
		network_export(ppm, EXPORT)

	# Plot de error de entrenamiento

	plt.plot(range(1, len(error_by_epoch[0])+1), error_by_epoch[0], marker='o', label="Training error")
	plt.plot(range(1, len(error_by_epoch[1])+1), error_by_epoch[1], marker='o', label="Training validation")

	plt.legend(loc='upper left')
	plt.annotate(error_by_epoch[0][-1], xy = (len(error_by_epoch[0]) + 1, error_by_epoch[0][-1]), bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
	plt.annotate(error_by_epoch[1][-1], xy = (len(error_by_epoch[0]) + 1, error_by_epoch[1][-1]), bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

	plt.xlabel('Epoch')
	plt.ylabel('Epoch Error')
	plt.show()
