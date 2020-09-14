import numpy as np
import cv2
import os
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface
from photometric_stereo import photometric_stereo as ps


file = './photometrics_images/MonkeyGray'

nbr_images = [5,25,45,65,85,105,121]

for n in nbr_images:
    ps(file, n, shadow_trick=False)
