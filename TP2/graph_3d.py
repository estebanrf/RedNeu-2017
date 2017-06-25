from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot(training, validation=[]):
	#los parametros de entrada deben ser [(categoria1,x1,y1,z1), ... , (categorian,xn, yn, zn)]

	colors=['b', 'g', 'r', 'pink', 'y', 'm', 'w', 'k', 'cyan']

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')


	for points, m in [(training, 'o'), (validation, '^')]:
		for cat, x, y, z in points:
			if isinstance(cat, int) or cat.is_integer():
				cat = int(cat)
			ax.scatter(x, y, z, c=colors[cat-1], marker=m)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.show()


#plot([(1,1.2,1.5,6.3),(1,2,3,2)], [(2,3,4,4), (3,4,5,9)])