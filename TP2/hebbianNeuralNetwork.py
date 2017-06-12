import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt( 'tp2_training_dataset.csv', delimiter=',')

class HebbianNeuralNetwork(object):


    def __init__(self, input_size, output_size):     

        self.w_ = np.random.rand(input_size, output_size)
        self.output_size = output_size

    def train(self, X, t=1000, is_sanger=true):

        while t > 0:
            eta = 1/t
            for x in X:  
                y = self.predict(x, (lambda x: 2*x))
                x_aprox = calculate_x_aproximation(y, is_sanger)               
                delta = eta * y * (x - x_aprox) 
                self.w_ += delta          	
                t = t - 1
        return self

    def net_input(self, x):

        return np.dot(x, self.w_)

    def predict(self, x, activation_function):
        
        return activation_function(self.net_input(x))

    def calculate_x_aproximation(self, y, is_sanger):
        x_aprox = []

        if is_sanger:
            for j in range(1, self.output_size + 1):
                x_aprox.append(np.multiply(y[0:j], self._w[0:j]).sum(axis=1))
        else:
            for j in range(1, self.output_size + 1):
                x_aprox.append(np.multiply(y[0:self.output_size], self._w[0: self.output_size]).sum(axis=1))
        return x_aprox