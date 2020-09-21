import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
from gauss2D import gauss2D
import sys
from scipy import signal
from math import sqrt

def compute_LoG(image, LOG_type):

    imOut = np.zeros(shape=image.shape)

    if LOG_type == 1:
        gaussian = signal.convolve2d(image,gauss2D(0.5,5), mode="same")
        imOut = signal.convolve2d(gaussian,np.array([[0,1,0],[1,-4,1],[0,1,0]]), mode="same")
        plt.imshow(imOut,cmap='gray')
        plt.title('Method 1: Smoothing by Gaussian followed by Laplacian')
        plt.axis('off')
        plt.show()
        print('hoi')

    elif LOG_type == 2:
        LoG = signal.convolve2d(gauss2D(0.5,5),np.array([[0,1,0],[1,-4,1],[0,1,0]]),mode="same")
        imOut = signal.convolve2d(image,LoG, mode="same")
        plt.imshow(imOut,cmap='gray')
        plt.title('Method 2: Directly with LoG kernel')
        plt.axis('off')
        plt.show()

    elif LOG_type == 3:
        DoG = gauss2D(0.5*np.sqrt(2),5)-gauss2D(0.5/np.sqrt(2),5)
        imOut = signal.convolve2d(image,DoG,mode="same")
        plt.imshow(imOut,cmap='gray')
        plt.title('Method 3: Difference of two Gaussians (DoG)')
        plt.axis('off')
        plt.show()

    return imOut

if __name__ == '__main__':
    print('hoi')
    img_path = 'images/image2.jpg'
    # Read with opencv
    I = cv2.imread(img_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    out_img = compute_LoG(I,int(sys.argv[1]))
