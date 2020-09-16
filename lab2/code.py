
## Imported by us for math operators, such as root or exp.
import numpy as np
import matplotlib.pyplot as plt # for visualizations
import Gabor_Segmentation.createGabor as cg

def  gauss1D(sigma ,kernel_size):
	# your code
	# compute ranges
	low = -np.floor(kernel_size / 2)
	high = np.floor(kernel_size / 2)
	x = np.arange(low, high+1, 1)
	
	# compute filter values
	G = np.exp(- (x ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

	# re-normalize
	G /= np.sum(G)

	return G

def  gauss2D(sigma ,kernel_size):
	# your code
	# compute 1D filters
	G_x = gauss1D(sigma, kernel_size)
	G_y = G_x.copy()

	# compute product and re-normalize
	G = np.outer(G_y, G_x)
	G /= np.sum(G)

	return G

def createGabor():
	# your code
	
	# dummy run
	sigma, theta, lamda, psi, gamma = (1,1,1,1,1)
	my_gabor = cg.createGabor(sigma, theta, lamda, psi, gamma)
	

if __name__ == "__main__":
	print('Testing gauss1D...')
	print('gauss1d(2,5) = ')
	print(gauss1D(2,5))

	print('Testing gauss2D...')
	print('gauss2d(2,5) = ')
	G_2d = gauss2D(2,10)
	print(G_2d)
	plt.imshow(G_2d)
	plt.show()

	print('Testing createGabor...')
	createGabor()