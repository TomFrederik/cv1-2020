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
        print('Loading images...\n')
        [image_stack, scriptV] = load_syn_images(image_dir, nfiles)
        [h, w, _, n] = image_stack.shape
        print('Finish loading %d images.\n' % n)

        colors = ['R', 'G', 'B']
        height_maps = default_dict(float)
        albedos = default_dict(float)
        SEs = default_dict(float)
        normals_dict = default_dict(float)
        
        for i in range(3):

            # compute the surface gradient from the stack of imgs and light source mat
            print('Computing surface albedo and normal map...\n')
            [albedo, normals] = estimate_alb_nrm(image_stack[:,:,i,:], scriptV, shadow_trick=shadow_trick)

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

        
        
    else:
        # obtain many images in a fixed view under different illumination
        print('Loading images...\n')
        [image_stack, scriptV] = load_syn_images(image_dir, nfiles)
        [h, w, n] = image_stack.shape
        print('Finish loading %d images.\n' % n)

        # compute the surface gradient from the stack of imgs and light source mat
        print('Computing surface albedo and normal map...\n')
        [albedo, normals] = estimate_alb_nrm(image_stack, scriptV, shadow_trick=False)

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
        show_results(albedo, normals, height_map, SE, n)

## Face
def photometric_stereo_face(image_dir='./yaleB02/'):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q )

    # show results
    show_results(albedo, normals, height_map, SE)
    
if __name__ == '__main__':
    photometric_stereo('./photometrics_images/SphereGray25/')
    #photometric_stereo_face()
