# import modules
import numpy as np
import pathlib as pt
from PIL import Image #needed for loading the image

def grey_world(image):
    '''
    Applies the grey world algorithm to an RBG image (represented as a numpy array)
    In:
    image - ndarray of shape (H, W, 3), the original image
    Out:
    clip_im - ndarray of shape (H, W, 3), image after applying the grey world algorithm and clipping from 0 to 255
    rescaled_image - ndarray of shape (H, W, 3), image after applying the GW algorithm and rescaling by max(im)/255
    '''

    # compute mean
    means = np.mean(image, axis=(0,1))
    
    ## we assume that the mean is given by 128*[alpha, beta, gamma], and that the image
    ## is given by [alpha*R_i, beta*G_i, gamma*B_i]
    
    # calculate coefficients [alpha, beta, gamma]
    coeffs = means/128
    

    # correct image by dividing out the illuminant coefficients
    corrected_im = image / coeffs[None, None, :]

    # clip from 0 to 255 OR rescale by multiplying with max/255 OR rescale channelwise
    clip_im = np.clip(corrected_im, 0, 255).astype(np.uint8)
    rescaled_im = (corrected_im / np.max(corrected_im) * 255).astype(np.uint8)
    rescaled_channel_im = (corrected_im / np.max(corrected_im, axis=(0,1)) * 255).astype(np.uint8)

    
    return clip_im, rescaled_im, rescaled_channel_im





if __name__ == "__main__":

    # specify file path
    dir_path = pt.Path.home() / 'Desktop/Projects/UvA_Period_1_1/CV_1/cv1_assignment_repo/lab1/colour_constancy'
    file_path = dir_path / 'awb.jpg'

    # load image and convert to numpy array of shape (320,256,3)
    image = Image.open(file_path)
    array_im = np.asarray(image)
    clip_arr, rescale_arr, rescale_channel_arr = grey_world(array_im)

    # convert to PIL images
    clip_im = Image.fromarray(clip_arr)
    rescale_im = Image.fromarray(rescale_arr)
    rescale_channel_im = Image.fromarray(rescale_arr)

    # save images
    clip_im.save(dir_path / 'awb_out_clip.jpg')
    rescale_im.save(dir_path / 'awb_out_rescale.jpg')
    rescale_channel_im.save(dir_path / 'awb_out_rescale_channel.jpg')