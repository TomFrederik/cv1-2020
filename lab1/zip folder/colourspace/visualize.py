def visualize(input_image):
    from getColourChannels import getColourChannels
    from matplotlib import pyplot as plt
    import numpy as np

    if(input_image.shape[2]==3):
        colourchannels = getColourChannels(input_image)
        fig=plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(input_image)
        plt.title('Converted Image')
        plt.subplot(2, 2, 2)
        plt.imshow(colourchannels[0],cmap='gray')
        plt.title('First Channel')
        plt.subplot(2, 2, 3)
        plt.imshow(colourchannels[1],cmap='gray')
        plt.title('Second Channel')
        plt.subplot(2, 2, 4)
        plt.imshow(colourchannels[2],cmap='gray')
        plt.title('Third Channel')
        plt.show()
    elif(input_image.shape[2]==4):
        fig=plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(input_image[:,:,0],cmap='gray')
        plt.title('Lightness')
        plt.subplot(2, 2, 2)
        plt.imshow(input_image[:,:,1],cmap='gray')
        plt.title('Average')
        plt.subplot(2, 2, 3)
        plt.imshow(input_image[:,:,2],cmap='gray')
        plt.title('Luminosity')
        plt.subplot(2, 2, 4)
        plt.imshow(input_image[:,:,3],cmap='gray')
        plt.title('OpenCV')
        plt.show()
    else:
        print("Error: unsupported shape of input_image")
