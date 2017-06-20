import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt( 'tp2_training_dataset.csv', delimiter=',')

class HebbianNeuralNetwork(object):


    def __init__(self, input_size, output_size):     

        self.w_ = np.random.rand(input_size, output_size)
        self.output_size = output_size

    def train(self, X, time_max=1000, is_sanger=True):
        t = 2
        while t < time_max:
            eta = 1/t
            for x in X:  
                y = self.predict(x[1:], (lambda y: 2*y))
                x_aprox = self.calculate_x_aproximation(y, is_sanger) 
                delta = eta * np.multiply(y, np.transpose((x[1:] - x_aprox))) 
                self.w_ += delta          	
                t = t + 1
        return self

    def net_input(self, x):

        return np.dot(x, self.w_)

    def predict(self, x, activation_function):
        
        return activation_function(self.net_input(x))

    def calculate_x_aproximation(self, y, is_sanger):
        x_aprox = []

        if is_sanger:
            for j in range(1, self.output_size + 1):
                x_aprox.append(np.multiply(y[0:j], self.w_[:,:j]).sum(axis=1))
                
        else:
            for j in range(1, self.output_size + 1):
                x_aprox.append(np.multiply(y[0:self.output_size], self.w_[:, :self.output_size]).sum(axis=1))
        return x_aprox


X = np.loadtxt( 'tp2_training_dataset.csv', delimiter=',')

hebbianSanger = HebbianNeuralNetwork(len(X[0])-1, 3)

hebbianSanger.train(X, 1000, True)

X_3d_Sanger = [ np.append([x[0]], hebbianSanger.predict(x[1:], (lambda x: 2*x))) for x in X]

hebbianOja = HebbianNeuralNetwork(len(X[0])-1, 3)

hebbianOja.train(X, 1000, True)

X_3d_Oja =  [ np.append([x[0]], hebbianSanger.predict(x[1:], (lambda x: 2*x))) for x in X]
