import numpy as np
import cv2
import os
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface


def get_albedo_norm(image_dir='./SphereGray5/', nfiles=None, shadow_trick=False):

    # obtain many images in a fixed view under different illumination
    print('Loading images...\n')
    [image_stack, scriptV] = load_syn_images(image_dir, nfiles)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)

    # compute the surface gradient from the stack of imgs and light source mat
    print('Computing surface albedo and normal map...\n')
    [albedo, normals] = estimate_alb_nrm(image_stack, scriptV, shadow_trick=shadow_trick)

    return albedo, normals

file_5 = './photometrics_images/SphereGray5'
file_25 = './photometrics_images/SphereGray25'

albedo_5_shad, norm_5_shad= get_albedo_norm(file_5, shadow_trick=True)
albedo_25_shad, norm_25_shad = get_albedo_norm(file_25, shadow_trick=True)

albedo_5, norm_5= get_albedo_norm(file_5)
albedo_25, norm_25 = get_albedo_norm(file_25)

alb_diff_5 = albedo_5 - albedo_5_shad
alb_diff_25 = albedo_25 - albedo_25_shad

norm_diff_5 = norm_5 - norm_5_shad
norm_diff_25 = norm_25 - norm_25_shad

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(alb_diff_5, vmin=-0.04, vmax=0.04)
plt.title('Albedo 5')
ax2 = fig.add_subplot(122)
im = ax2.imshow(alb_diff_25, vmin=-0.04, vmax=0.04)
fig.colorbar(im)
plt.title('Albedo 25')
plt.savefig('./shadow_trick_investigation/Albedo_difference.pdf')

# showing normals as three separate channels
figure = plt.figure()
ax1 = figure.add_subplot(231)
ax1.imshow(norm_diff_5[..., 0], vmin=0, vmax=0.4)
ax2 = figure.add_subplot(232)
ax2.imshow(norm_diff_5[..., 1], vmin=0, vmax=0.4)
ax3 = figure.add_subplot(233)
im = ax3.imshow(norm_diff_5[..., 2], vmin=0, vmax=0.4)
figure.colorbar(im)
plt.title('Norm 5')
ax1 = figure.add_subplot(234)
ax1.imshow(norm_diff_25[..., 0], vmin=0, vmax=0.4)
ax2 = figure.add_subplot(235)
ax2.imshow(norm_diff_25[..., 1], vmin=0, vmax=0.4)
ax3 = figure.add_subplot(236)
im = ax3.imshow(norm_diff_25[..., 2], vmin=0, vmax=0.4)
figure.colorbar(im)
plt.title('Norm 25')
plt.savefig('./shadow_trick_investigation/Norm_difference.pdf')


