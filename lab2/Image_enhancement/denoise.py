import cv2
from myPSNR import myPSNR
import numpy as np
import matplotlib.pyplot as plt

def denoise( image, kernel_type, **kwargs):
    if kernel_type == 'box':
        imOut = cv2.blur(image, (kwargs['kernel_size'], kwargs['kernel_size']))
    elif kernel_type == 'median':
        imOut = cv2.medianBlur(image, kwargs['kernel_size'])
    elif kernel_type == 'gaussian':
        imOut = cv2.GaussianBlur(image, kwargs['kernel_size'], sigma) # use our function here? Instructions unclear.
    else:
        print('Operation Not implemented')
        return None
    return imOut

if __name__ == "__main__":
    out_dir = './q7_results/'
    img = cv2.imread('./images/image1.jpg')
    sp_img = cv2.imread('./images/image1_saltpepper.jpg')
    gauss_img = cv2.imread('./images/image1_gaussian.jpg')
    plt.show()
    psnrs = np.zeros((4,3))

    for i,kernel_size in enumerate([3,5,7]):
        kwargs = {'kernel_size':kernel_size}

        print('Denoising salt_pepper image with [{0}x{0}] Box filter..'.format(kernel_size))
        out_im = denoise(sp_img, 'box', **kwargs)
        plt.figure()
        plt.imshow(out_im)
        plt.savefig(out_dir + 'box_saltpepper_{}.pdf'.format(kernel_size))
        psnrs[0,i] = myPSNR(img, out_im)

        print('Denoising gaussian image with [{0}x{0}] Box filter..'.format(kernel_size))
        out_im = denoise(gauss_img, 'box', **kwargs)
        plt.figure()
        plt.imshow(out_im)
        plt.savefig(out_dir + 'box_gaussian_{}.pdf'.format(kernel_size))
        psnrs[1,i] = myPSNR(img, out_im)

        print('Denoising salt_pepper image with [{0}x{0}] Median filter..'.format(kernel_size))
        out_im = denoise(sp_img, 'median', **kwargs)
        plt.figure()
        plt.imshow(out_im)
        plt.savefig(out_dir + 'median_saltpepper_{}.pdf'.format(kernel_size))
        psnrs[2,i] = myPSNR(img, out_im)

        print('Denoising gaussian image with [{0}x{0}] Median filter..'.format(kernel_size))
        out_im = denoise(gauss_img, 'median', **kwargs)
        plt.figure()
        plt.imshow(out_im)
        plt.savefig(out_dir + 'median_gaussian_{}.pdf'.format(kernel_size))
        psnrs[3,i] = myPSNR(img, out_im)
    
    print('PSNRs = ',psnrs)
    
    # write psnrs to file
    file = open(out_dir + 'box_median_psnr.txt', 'w')
    file.write('kernel_size         =   3   |   5   |   7  \n')
    file.write('------------------------------------------\n')
    file.write('PSNR Box (SP)       = {0:1.2f} | {1:1.2f} | {2:1.2f}\n'.format(*tuple([psnrs[0,i] for i in range(3)])))
    file.write('PSNR Box (Gauss)    = {0:1.2f} | {1:1.2f} | {2:1.2f}\n'.format(*tuple([psnrs[1,i] for i in range(3)])))
    file.write('PSNR Median (SP)    = {0:1.2f} | {1:1.2f} | {2:1.2f}\n'.format(*tuple([psnrs[2,i] for i in range(3)])))
    file.write('PSNR Median (Gauss) = {0:1.2f} | {1:1.2f} | {2:1.2f}'.format(*tuple([psnrs[3,i] for i in range(3)])))
    file.close() 
