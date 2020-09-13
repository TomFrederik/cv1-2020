import numpy as np
import cv2
import os
from collections import * # for default dict
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface

print('Part 1: Photometric Stereo\n')

def photometric_stereo(image_dir='./SphereGray5/', nfiles=None, color=False, shadow_trick=True):

    if color:
        
        colors = ['R', 'G', 'B']
        height_maps = defaultdict(float)
        albedos = defaultdict(float)
        SEs = defaultdict(float)
        normals_dict = defaultdict(float)
        not_ignore = defaultdict(float)

        for i in range(3):
            print('Loading images for {} channel\n'.format(colors[i]))
            [image_stack, scriptV] = load_syn_images(image_dir, nfiles, channel=i)
            [h, w, n] = image_stack.shape
            print('Finish loading %d images.\n' % n)
    
            if not shadow_trick:
                raise Warning('Shadow-trick disabled. This is recommended for RGB images!')
            
            # save pixel indices that are zero for all images in this channel. They will be ignored
            # when averaging the height map
            not_ignore[colors[i]] = np.sum(image_stack, axis=-1) != 0
            print(not_ignore[colors[i]].shape)
            # compute the surface gradient from the stack of imgs and light source mat
            print('Computing surface albedo and normal map...\n')
            [albedo, normals] = estimate_alb_nrm(image_stack, scriptV, shadow_trick=shadow_trick)

            # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
            print('Integrability checking\n')
            [p, q, SE] = check_integrability(normals)

            threshold = 0.005;
            print('Number of outliers: %d\n' % np.sum(SE > threshold))
            SE[SE <= threshold] = float('nan') # for good visualization

            # compute the surface height
            print('Computing the height map...')
            height_map = construct_surface( p, q,'average' )

            # add to the dicts
            height_maps[colors[i]] = height_map
            albedos[colors[i]] = albedo
            SEs[colors[i]] = SEs
            normals_dict[colors[i]] = normals
        
        #####
        # NOT SURE HOW TO COMBINE YET
        #####
        # averaging the height maps:
        height_map = np.zeros_like(height_maps[colors[0]])
        inc_mean_ctr = np.ones_like(height_map)
        for i in range(3):
            height_map[not_ignore[colors[i]]] += 1/inc_mean_ctr[not_ignore[colors[i]]] \
                                                * (height_maps[colors[i]][not_ignore[colors[i]]] \
                                                  - height_map[not_ignore[colors[i]]])
            inc_mean_ctr[not_ignore[colors[i]]] += 1
        
        # plotting the height map
        stride = 1
        X, Y = np.meshgrid(np.arange(0,np.shape(normals_dict[colors[0]])[0], stride),
        np.arange(0,np.shape(normals_dict[colors[0]])[1], stride))
    
        
        H = height_map[::stride,::stride]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, H.T)
        ax.set_zlim(bottom=0,top=512)
        plt.savefig('./reconstruction_results/SphereColor_height.pdf')
        
        
    else:
        # obtain many images in a fixed view under different illumination
        print('Loading images...\n')
        [image_stack, scriptV] = load_syn_images(image_dir, nfiles)
        [h, w, n] = image_stack.shape
        print('Finish loading %d images.\n' % n)

        # compute the surface gradient from the stack of imgs and light source mat
        print('Computing surface albedo and normal map...\n')
        [albedo, normals] = estimate_alb_nrm(image_stack, scriptV, shadow_trick=shadow_trick)

        # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
        print('Integrability checking\n')
        [p, q, SE] = check_integrability(normals)

        threshold = 0.005;
        print('Number of outliers: %d\n' % np.sum(SE > threshold))
        SE[SE <= threshold] = float('nan') # for good visualization

        # compute the surface height
        print('Computing the height map...')
        height_map = construct_surface( p, q,'average' )

        # show results
        print('Showing results...')
        show_results(albedo, normals, height_map, SE)

## Face
def photometric_stereo_face(image_dir='./photometrics_images/yaleB02/'):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV, shadow_trick=False)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q )

    # show results
    show_results(albedo, normals, height_map, SE, zlim=180)
    
if __name__ == '__main__':
    photometric_stereo('./photometrics_images/SphereColor/', color=True)
    #photometric_stereo_face()
