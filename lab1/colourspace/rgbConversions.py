import numpy as np
import cv2
from getColourChannels import getColourChannels

def rgb2grays(input_image):
    new_image=np.array(input_image)
    c = getColourChannels(new_image)
    new_image = np.empty((input_image.shape[0],input_image.shape[1],4), dtype=np.float32)
    # converts an RGB into grayscale by using 4 different methods

    # ligtness method
    new_image[:, :, 0] = (np.max(input_image,axis=2)+np.min(input_image,axis=2))/2
    # average method
    new_image[:, :, 1] = (c[0]+c[1]+c[2])/3
    # luminosity method
    new_image[:, :, 2] = 0.21*c[0] + 0.72*c[1] + 0.07*c[2]
    # built-in opencv function
    new_image[:, :, 3] = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    new_image=np.array(input_image)
    colourchannels = getColourChannels(new_image)
    input_image[:, :, 0] = (colourchannels[0]-colourchannels[1])/np.sqrt(2)
    input_image[:, :, 1] = (colourchannels[0]+colourchannels[1]-2*colourchannels[2])/np.sqrt(6)
    input_image[:, :, 2] = (colourchannels[0]+colourchannels[1]+colourchannels[2])/np.sqrt(3)

    return input_image


def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space
    new_image=np.array(input_image)
    colourchannels = getColourChannels(new_image)
    denominator = colourchannels[0]+colourchannels[1]+colourchannels[2]
    input_image[:, :, 0] = colourchannels[0]/denominator
    input_image[:, :, 1] = colourchannels[1]/denominator
    input_image[:, :, 2] = colourchannels[2]/denominator

    return input_image
