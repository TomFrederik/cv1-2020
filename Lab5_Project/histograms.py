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

    sift_des = []

    # get the first 50 images of each class
    (_,_, files) = next(os.walk(folder))
    files = [folder + f for f in files][50:150]

    # get Keypoint SIFT descriptors for each image
    for cur_file in files:
        img = cv.imread(cur_file)
        _, des = sift.detectAndCompute(img, None)
        
        for j in range(des.shape[0]):
            sift_des.append(des[j,:])


    sift_des = np.array(sift_des)


    # compute distance to clusters
    distance = np.sum((sift_des[:,None,:] - dictionary[None,:]) ** 2, axis=-1)

    # assign each datapoint to the min distance cluster
    cluster_assignments = np.argmin(distance, axis=-1)

    # plot hist
    plt.figure()
    plt.hist(cluster_assignments, bins=1000, density=True)
    plt.xlabel('Visual Word')
    plt.ylabel('Frequency')
    plt.savefig('./histograms/'+class_names[i]+'.pdf')




