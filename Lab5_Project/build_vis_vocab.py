import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt


class_names = ['airplane', 'bird', 'ship', 'horse', 'car']
class_nbrs = [1,2,3,4,5]

DATA_DIR = './data/train/'
class_dirs = [DATA_DIR + str(nbr) + '/' for nbr in class_nbrs]


dictionary = np.load('./dictionaries/k_{}.npy'.format(4000))



# compute descriptors for 50 images of each class



# perform sift to get keypoints and compute descriptors
sift = cv.xfeatures2d.SIFT_create()


for i, folder in enumerate(class_dirs):

    class_vocab = np.zeros((100, 4000))

    # get the first 50 images of each class
    (_,_, files) = next(os.walk(folder))
    files = [folder + f for f in files][50:150]

    # get Keypoint SIFT descriptors for each image
    for j, cur_file in enumerate(files):
        img = cv.imread(cur_file)
        _, des = sift.detectAndCompute(img, None)

        # compute distance to clusters
        distance = np.sum((des[:,None,:] - dictionary[None,:]) ** 2, axis=-1)

        # assign each datapoint to the min distance cluster
        cluster_assignments = np.argmin(distance, axis=-1)

        # inc counter of word for this image
        for idx in cluster_assignments:
            class_vocab[j,idx] += 1 



    np.save('./vocabulary/train_'+class_names[i]+'.npy', class_vocab)

    

    
