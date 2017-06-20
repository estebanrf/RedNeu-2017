import numpy as np
import matplotlib.pyplot as plt
import random
import math
MINIMUN_COLOUR_DENSITY = 0.01

def cell(category_count):
    return np.zeros((category_count, 2))

class CompetitiveNeuralNetwork(object):

    def __init__(self, input_size, n, m, category_count, neurons):

        # self.neurons = np.random.uniform(0.5, 20, (n*m, input_size))
        self.neurons = neurons
        print len(neurons)
        self.map = np.array([cell(category_count) for _ in range(n) for _ in range(m)])
        self.input_size = input_size
        self.n = n
        self.m = m

    # tau1 = contante de tiempo elegida para decrementar el area de a vecindad en el tiempo
    # tau2 = constante de tiempo elegida para decrementar el factor de aprendizaje en el tiempo
    # alfa = es el eta, el factor de aprendizaje
    # sigma = es el ancho de la campana de gauss
    def train(self, X, is_scalar_product=False, tau1=(1000/math.log(5)), alfa=0.01, tau2=1000, sigma=7, iterations=1000):
        sigma0 = sigma
        alfa0 = alfa
        for t in range(iterations):
            x = random.choice(X)
            winner_index = self.find_winner_neuron(x[1:], is_scalar_product)
            self.update_winner_and_neighbors(winner_index, x, alfa, sigma, is_scalar_product)
            sigma = sigma0*math.exp(-(t/tau1))
            alfa = alfa0*math.exp(-(t/tau2))

        return self.map

    def plot_category(self, map_category):
      colors = []
      for cell in map_category:
        category_max = cell[0]
        if (cell[0][1] == 0):
          category_color = 0
        else:
          category_color = cell[0][0]/cell[0][1]
        for idx, category in zip(range(len(cell)),cell):
          if(category_max[0]/category_max[1] <= category[0]/category[1]):
            category_max = category
            category_color = idx + (category[0]/category[1])
        colors.append(category_color)
        print category_color

      all_colors = np.zeros((30, 30))
      for i in range(30):
        for j in range(30):
          all_colors[i, j] = colors[i * 30 + j]

      return all_colors

    def calculate_mexican_gaussian(self, sigma, winner_i_j, current_i_j):
        return math.exp(-(math.pow(np.linalg.norm(np.subtract(winner_i_j, current_i_j)), 2) / (2 * math.pow(sigma,2))))

    def update_winner_and_neighbors(self, winner, x, alfa, sigma, is_scalar_product):
        winner_i_j = [winner / self.m, winner % self.m]
        for i in range(self.n):
            for j in range(self.m):
                gaussian_factor = self.calculate_mexican_gaussian(sigma, winner_i_j ,[i,j])
                print gaussian_factor
                if gaussian_factor > MINIMUN_COLOUR_DENSITY:
                  self.map[i * self.m + j][int(x[0])-1] += [gaussian_factor, 1]

                delta = alfa * gaussian_factor * (x[1:] - self.neurons[winner])

                self.neurons[i * self.m + j] = np.add(self.neurons[i * self.m + j], delta)
                if is_scalar_product:
                  self.neurons[i * self.m + j] /= np.linalg.norm(self.neurons[i * self.m + j])

    def find_winner_neuron(self, x, is_scalar_product):
        if is_scalar_product:
          minimum_distance_index = np.argmin(np.dot(x, np.transpose(self.neurons)))
          return minimum_distance_index
        else:
          minimum_distance_index = np.argmin(np.linalg.norm(self.neurons - x, axis = 1))
          return minimum_distance_index


X = np.loadtxt( 'tp2_training_dataset.csv', delimiter=',')
competitveNeuralNetwork = CompetitiveNeuralNetwork(len(X[0])-1, 30, 30, 9, X[:(30*30), 1:])
map_category = competitveNeuralNetwork.train(X, True, (1000/math.log(5)), 0.5, 1000, 5, 1000)
map_result = competitveNeuralNetwork.plot_category(map_category)

plt.matshow(map_result)
plt.show()
print("termine")
