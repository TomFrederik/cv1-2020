import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import signal

def compute_gradient(image):

    Gx_kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Gx = signal.convolve2d(image,Gx_kernel,mode="same")
    Gy_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Gy = signal.convolve2d(image,Gy_kernel,mode="same")
    im_magnitude = np.sqrt(Gx**2 + Gy**2)
    im_direction = np.arctan(Gy/Gx)

    print(image.shape)
    print(Gx.shape)
    fig=plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(image,cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(Gx,cmap='gray')
    plt.title('Gradient X Direction')
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.imshow(Gy,cmap='gray')
    plt.title('Gradient Y Direction')
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.imshow(im_magnitude, cmap="gray")
    plt.title('Gradient Magnitude')
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(im_direction, cmap="gray")
    plt.title('Gradient Direction')
    plt.axis('off')
    plt.show()

    return Gx, Gy, im_magnitude,im_direction


if __name__ == '__main__':
    img_path = 'images/image2.jpg'
    # Read with opencv
    I = cv2.imread(img_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    out_img = compute_gradient(I)
