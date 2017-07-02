import numpy as np
from graph_3d import plot
import re

def main():

    def filter_by_categories(X, cats):
        return [t for t in X if t[0] in cats]

    def center(x):
        avg = np.mean(x, axis=0)

        return x - avg

    class HebbianNeuralNetwork(object):

        def __init__(self, input_size, output_size):

            self.w_ = np.random.rand(input_size, output_size)
            self.output_size = output_size

        def train(self, X, time_max=1000, is_sanger=True):
            t = 2
            X = center(X)
            while t < time_max:
                eta = .001
                for x in X:
                    y = self.predict(x[1:], (lambda d: 2*d))
                    x_aprox = self.calculate_x_aproximation(y, is_sanger)
                    delta = eta * np.multiply(y, np.transpose(x[1:] - x_aprox))
                    self.w_ += delta
                    t += 1
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

    X = np.loadtxt('tp2_training_dataset.csv', delimiter=',')
    X_tr = X[:int(len(X)*0.9)]
    X_va = X[int(len(X)*0.9):]

    use_sanger = False
    output_size = 9
    iterations = 5000

    if True: #para entrenar o importar
        hebbian = HebbianNeuralNetwork(len(X_tr[0]) - 1, output_size)
        hebbian.train(X_tr, iterations, use_sanger)

        if False: #para exportar
            np.savetxt('heb1.txt', hebbian.w_.flatten(), header=str(hebbian.w_.shape))

    else: #importa
        with open('heb1.txt') as f:
            shape = tuple(int(num) for num in re.findall(r'\d+', f.readline()))
        w_ = np.loadtxt('heb1.txt').reshape(shape)
        hebbian = HebbianNeuralNetwork(len(X_tr[0]) - 1, output_size)
        hebbian.w_ = w_

    X_3d_tr = [np.append([x[0]], hebbian.predict(x[1:], (lambda x: 2*x))) for x in X_tr]
    X_3d_va = [np.append([x[0]], hebbian.predict(x[1:], (lambda x: 2*x))) for x in X_va]

    # X_3d = filter_by_categories(X_3d, [1,2,3])
    # plot(X_3d_tr, X_3d_va)

    return X_3d_tr
