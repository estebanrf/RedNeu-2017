from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot(training, validation):
	#los parametros de entrada deben ser [(x1,y1,z1,categoria1), ... , (xn, yn, zn,categoria9)]

	colors=['b', 'g', 'r', 'pink', 'y', 'm', 'w', 'k', 'cyan']

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')


	for points, m in [(training, 'o'), (validation, '^')]:
		for x, y, z, cat in points:
		    ax.scatter(x, y, z, c=colors[cat-1], marker=m)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.show()


plot([(1,1,1,6),(1,2,3,2)], [(2,3,4,4), (3,4,5,9)])