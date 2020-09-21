import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
from gauss2D import gauss2D
import sys
from scipy import signal
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
from gauss2D import gauss2D
import sys
from scipy import signal
from math import sqrt


def L(s,x,y): 
    '''
    compute the LoG at a specific location in space
    '''

    return(-1/(np.pi*s**4) * np.exp(-1/(2 * s**2 ) * (x**2 + y**2)) * (1 - (x**2 + y**2) / (2*s**2)))


def compute_LoG_kernel(n,s):
    '''
    computes 'exact' nxn LoG with std s centered around [n/2,n/2]
    '''
    zero = int((n-1)/2)

    out = np.zeros((n,n))

    for i in range(-zero,zero+1,1):
        for j in range(-zero,zero+1,1):
            out[i+zero,j+zero] = L(s,i,j)
    return out


def compute_LoG(image, LOG_type):

    imOut = np.zeros(shape=image.shape)

    if LOG_type == 1:
        gaussian = signal.convolve2d(image,gauss2D(0.5,5), mode="same")
        imOut = cv2.Laplacian(gaussian, ddepth=-1)
        plt.imshow(imOut,cmap='gray')
        plt.title('Method 1: Smoothing by Gaussian followed by Laplacian')
        plt.axis('off')
        #plt.show()
        plt.savefig('./log_results/method_1.pdf')

    elif LOG_type == 2:
        # computing the LoG kernel from scratch
        LoG_exact = compute_LoG_kernel(5,0.5)
        
        # take LoG kernel from internet
        #Log_internet = $$$
        
        # deprecated, not correct
        #LoG = signal.convolve2d(gauss2D(0.5,5),np.array([[0,1,0],[1,-4,1],[0,1,0]]),mode="same")
        
        # compute for exact kernel
        imOut = signal.convolve2d(image,LoG_exact, mode="same")
        # compute for internet kernel
        #imOut = signal.convolve2d(image,LoG_internet, mode="same")
        
        plt.imshow(imOut,cmap='gray')
        plt.title('Method 2: Directly with LoG kernel')
        plt.axis('off')
        plt.savefig('./log_results/method_2.pdf')
        #plt.show()

    elif LOG_type == 3:
        DoG = gauss2D(0.5*np.sqrt(2),5)-gauss2D(0.5/np.sqrt(2),5)
        imOut = signal.convolve2d(image,DoG,mode="same")
        plt.imshow(imOut,cmap='gray')
        plt.title('Method 3: Difference of two Gaussians (DoG)')
        plt.axis('off')
        plt.savefig('./log_results/method_3.pdf')
        #plt.show()


    return imOut

if __name__ == '__main__':
    img_path = 'images/image2.jpg'
    # Read with opencv
    I = cv2.imread(img_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)


    for log_type in (1,2,3):
        out_im = compute_LoG(I, log_type)
        print(out_im.shape)
    
