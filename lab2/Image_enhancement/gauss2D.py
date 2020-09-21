from gauss1D import gauss1D
import numpy as np

def gauss2D( sigma , kernel_size ):
    ## solution
    # compute 1D filters
	G_x = gauss1D(sigma, kernel_size)
	G_y = G_x.copy()

	# compute product and re-normalize
	G = np.outer(G_y, G_x)
	G /= np.sum(G)

	return G
