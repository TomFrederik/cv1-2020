import numpy as np

def gauss1D( sigma , kernel_size )
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    # solution
	# compute ranges
	low = -np.floor(kernel_size / 2)
	high = np.floor(kernel_size / 2)
	x = np.arange(low, high+1, 1)
	
	# compute filter values
	G = np.exp(- (x ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

	# re-normalize
	G /= np.sum(G)
    
	return G
