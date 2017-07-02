import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import re
from hebbian import main

class CompetitiveNeuralNetwork(object):

    def __init__(self, input_size, n, m, category_count, neurons):

        self.neurons = neurons
        self.map = np.zeros((n*m, category_count))
        self.input_size = input_size
        self.n = n
        self.m = m

    # tau1 = contante de tiempo elegida para decrementar el area de a vecindad en el tiempo
    # tau2 = constante de tiempo elegida para decrementar el factor de aprendizaje en el tiempo
    # eta = es el eta, el factor de aprendizaje
    # sigma = es el ancho de la campana de gauss
    def train(self, X, is_scalar_product=False, sigma=7, tau1=(1000/math.log(5)), eta=0.01, tau2=1000, iterations=1000):
        sigma0 = sigma
        eta0 = eta
        for t in range(iterations):
            print "iteracion: "
            print t
            for x in X[:720]:
                winner_index = self.find_winner_neuron(x[1:], is_scalar_product)
                self.map[winner_index][int(x[0]) - 1] += 1
                self.update_winner_and_neighbors(winner_index, x, eta, sigma, is_scalar_product)
            sigma = sigma0*math.exp(-(t/tau1))
            eta = eta0*math.exp(-(t/tau2))

        map_result = self.calculate_map(self.map)
        asserts = 0
        for x in X[720:]:
          winner_index = self.find_winner_neuron(x[1:], is_scalar_product)
          winner_i_j = np.array([winner_index / self.m, winner_index % self.m])
          if map_result[winner_i_j[0]][winner_i_j[1]] == x[0]:
            asserts += 1
        print asserts/90.0
        return map_result

    def calculate_map(self, map_category):
        map_result = np.zeros((self.n, self.m))
        for cell, idx in zip(map_category, range(self.n * self.m)):
            if np.sum(cell) == 0:
              category_max_idx = -1
            category_max_idx = np.argmax(cell)
            map_result[idx/self.m, idx % self.m] = category_max_idx + 1

        return map_result

    def calculate_mexican_gaussian(self, sigma, winner_i_j, current_i_j):
        distance = math.pow(np.linalg.norm(current_i_j - winner_i_j), 2)
        c = 2 * math.pow(sigma,2)
        a = sigma * math.sqrt(2*math.pi)
        return math.exp(-distance/c)/a

    def update_winner_and_neighbors(self, winner, x, eta, sigma, is_scalar_product):
        winner_i_j = np.array([winner / self.m, winner % self.m])
        for i in range(self.n):
            for j in range(self.m):
                gaussian_factor = self.calculate_mexican_gaussian(sigma, winner_i_j ,np.array([i,j]))
                delta = eta * gaussian_factor * (x[1:] - self.neurons[winner])
                self.neurons[i * self.m + j] += delta
                # if is_scalar_product:
                self.neurons[i * self.m + j] /= np.linalg.norm(self.neurons[i * self.m + j])

    def find_winner_neuron(self, x, is_scalar_product):
        if is_scalar_product:
            minimum_distance_index = np.argmin(np.dot(x, np.transpose(self.neurons)))
            return minimum_distance_index
        else:
            minimum_distance_index = np.argmin(np.linalg.norm(self.neurons - x, ord=2, axis = 1))
            return minimum_distance_index

X = main()
X = np.resize(X, (len(X), len(X[0])))
Y = X
random.shuffle(Y)
competitveNeuralNetwork = CompetitiveNeuralNetwork(len(X[0])-1, 10, 10, 9, Y[:(10*10), 1:])
# X, is_scalar_product=False, sigma=7, tau1=(1000/math.log(5)), eta=0.01, tau2=1000, iterations=1000):
map_result = competitveNeuralNetwork.train(X, False, 10.0, (150.0/math.log(10)), 0.1, 150.0, 150)

plt.matshow(map_result)
plt.show()
