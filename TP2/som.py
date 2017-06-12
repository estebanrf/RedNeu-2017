import numpy as np
import matplotlib.pyplot as plt
import random
import math

MINIMUN_COLOUR_DENSITY = 0.01

def normalizedRandomNeuron(input_size):
    neuron = np.random.uniform(0.5, 20, input_size)
    neuron /= np.linalg.norm(neuron)
    return [[[],[],[],[],[],[],[],[],[]], neuron]

class CompetitiveNeuralNetwork(object):

    def __init__(self, input_size, n, m):     
        
        self.map = [[normalizedRandomNeuron(input_size) for _ in range(m)] for _ in range(n)]
        self.input_size = input_size
        self.n = n
        self.m = m

    def train(self, X, is_scalar_product=False, tau1=(1000/math.log(5)), alfa=0.1, tau2=1000, sigma=5, iterations=1000):
        
        t = 1
        alfa0 = alfa
        sigma0 = sigma
        
        while t < iterations:
            
            x = random.choice(X)
            winner_i_j = self.find_winner_neuron(x[1:], is_scalar_product)
            
            self.update_winner_and_neighbors(winner_i_j, x, alfa, sigma, is_scalar_product)
    
            sigma = sigma0*math.exp(-(t/tau1))
            alfa = alfa0*math.exp(-(t/tau2))        
            t = t + 1
        return self
    
    def calculate_mexican_gaussian(self, sigma, winner_i_j, current_i_j):
        return math.exp(-(math.pow(np.linalg.norm(np.subtract(winner_i_j,current_i_j)/sigma),2)/2))

    def update_winner_and_neighbors(self, winner_i_j, x, alfa, sigma, is_scalar_product):
        for i in range(0, self.n):
            for j in range(0, self.m):
                gaussian_factor = self.calculate_mexican_gaussian(sigma, winner_i_j ,[i,j])
                
                if gaussian_factor > MINIMUN_COLOUR_DENSITY:
                    self.map[i][j][0][int(x[0])-1].append(gaussian_factor)

                delta = alfa * gaussian_factor * (x[1:] - self.map[i][j][1])

                self.map[i][j][1] = np.add(self.map[i][j][1], delta)
                if is_scalar_product:
                    self.map[i][j][1] /= np.linalg.norm(self.map[i][j][1])             
        
    def find_winner_neuron(self, x, is_scalar_product):
        #Despues se refactoriza.
        winner_neuron = [0,0]
        if is_scalar_product:
            maximun_value = self.map[0][0][1] * x
            for i in range(0, self.n):
                for j in range(0, self.m):
                    current_value = self.map[i][j][1] * x
                    if current_value > maximun_value:
                        maximun_value = current_value
                        winner_neuron = [i,j]  
        else:
            minimum_distance = np.linalg.norm(x - self.map[0][0][1])
            for i in range(0, self.n):
                for j in range(0, self.m):
                    current_distance =  np.linalg.norm(x - self.map[i][j][1])
                    if current_distance < minimum_distance:
                        minimum_distance = current_distance
                        winner_neuron = [i,j]
        return winner_neuron


X = np.loadtxt( 'tp2_training_dataset.csv', delimiter=',')

competitveNeuralNetwork = CompetitiveNeuralNetwork(len(X[0])-1, 20, 20)

competitveNeuralNetwork.train(X, False, (1000/math.log(5)), 0.1, 1000, 5, 1000)

print("termine")