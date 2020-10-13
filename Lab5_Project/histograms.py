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
    # Becase the full array of pairwise distances takes about 10 GiB,
    # we chunk the sum into several batches
    sum_len = sift_des.shape[-1]
    per_chunk = 10
    print("Number of chunks:", sum_len // per_chunk + 1)

    distance = np.empty((sift_des.shape[0], dictionary.shape[0]))
    for l in range(sum_len // per_chunk + 1):
        indices = slice(l * per_chunk, (l + 1) * per_chunk)
        distance += np.sum((sift_des[:,None, indices] - dictionary[None,:, indices]) ** 2, axis=-1)

    # assign each datapoint to the min distance cluster
    cluster_assignments = np.argmin(distance, axis=-1)
    np.save('./histogram_data/cluster_assignments_{}.npy'.format(i + 1), cluster_assignments)

    # plot hist
    plt.figure()
    plt.hist(cluster_assignments, bins=1000, density=True)
    plt.xlabel('Visual Word')
    plt.ylabel('Frequency')
    plt.savefig('./histograms/'+class_names[i]+'.pdf')




