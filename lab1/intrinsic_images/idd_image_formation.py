import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def main():    
    # int32 to prevent overflow of the standard uint8 during element-wise multiplication
    original = np.array(Image.open('ball.png'))
    shading = np.array(Image.open('ball_shading.png'), dtype="int32")
    albedo = np.array(Image.open('ball_albedo.png'), dtype="int32")

    # I(x)=R(X) x S(x) for all x in one line
    # divide by 255 to renormalise
    # dividing by 256 gives 0 RMS
    reconstruction_255 = np.array(albedo*shading[:,:,None]/255,dtype="int32")
    reconstruction_256 = np.array(albedo*shading[:,:,None]/256,dtype="int32")
    print('RMS_255 = {0:1.4f}'.format(np.sqrt(np.mean((reconstruction_255-original)**2))))
    print('RMS_256 = {0:1.4f}'.format(np.sqrt(np.mean((reconstruction_256-original)**2))))

    # plotting
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(shading,cmap='gray',vmin=0,vmax=255)
    plt.title('Shading')
    plt.subplot(2, 2, 3)
    plt.imshow(albedo)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Albedo')
    plt.subplot(2, 2, 4)
    plt.imshow(reconstruction_255)
    plt.title('Reconstruction')    
    plt.savefig('image_reconstruction.pdf')

if __name__ == "__main__":
    main()
