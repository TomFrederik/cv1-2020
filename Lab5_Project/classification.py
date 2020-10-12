import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt


class_names = ['airplane', 'bird', 'ship', 'horse', 'car']
class_nbrs = [1,2,3,4,5]

DATA_DIR = './data/train/'
class_dirs = [DATA_DIR + str(nbr) + '/' for nbr in class_nbrs]


dictionary = np.load('./dictionaries/k_{}.npy'.format(4000))

## TODO: Finish this