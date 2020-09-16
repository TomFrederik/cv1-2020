# import numpy for math operators
import numpy as np
import PIL.Image as im # for loading the image


def myPSNR( orig_image, approx_image ):

    # compute maximum and RMSE
    I_max = np.max(orig_image)
    RMSE = np.sqrt(np.mean((orig_image - approx_image) ** 2))
    
    # compute PSNR
    PSNR = 20 * np.log10(I_max / RMSE)
    
    return PSNR

if __name__ == "__main__":
    
    print('Computing myPSNR between image1_saltpepper and image1...')
    # load images
    image1 = np.array(im.open('./images/image1.jpg'))
    image1_saltpepper = np.array(im.open('./images/image1_saltpepper.jpg'))

    # compute PSNR
    PSNR_saltpepper_1 = myPSNR(image1, image1_saltpepper)
    print('PSNR = {0:1.3f}'.format(PSNR_saltpepper_1))

    print('Computing myPSNR between image1_gaussian and image1...')
    # load images
    image1 = np.array(im.open('./images/image1.jpg'))
    image1_gaussian = np.array(im.open('./images/image1_gaussian.jpg'))
    
    # compute PSNR
    PSNR_gaussian_1 = myPSNR(image1, image1_gaussian)
    print('PSNR = {0:1.3f}'.format(PSNR_gaussian_1))