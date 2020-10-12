import numpy as np
import cv2 as cv
import os
import scipy.cluster.vq as vq


class_names = ['airplane', 'bird', 'ship', 'horse', 'car']
class_nbrs = [1,2,3,4,5]

DATA_DIR = './data/train/'
class_dirs = [DATA_DIR + str(nbr) + '/' for nbr in class_nbrs]

sift_des = []

# perform sift to get keypoints and compute descriptors
sift = cv.xfeatures2d.SIFT_create()


for i, folder in enumerate(class_dirs):
    # get the first 100 images of each class
    (_,_, files) = next(os.walk(folder))

    files = [folder + f for f in files][:50]

    # get Keypoint SIFT descriptors for each image
    for cur_file in files:
        img = cv.imread(cur_file)
        _, des = sift.detectAndCompute(img, None)
        
        for j in range(des.shape[0]):
            sift_des.append(des[j,:])


sift_des = np.array(sift_des)
print('SIFT des shape:', sift_des.shape)

# now perform K-means clustering on these descriptors
num_clusters = [400, 1000, 4000]
iters = 20
threshold = 1e-5

for k in num_clusters:
    codebook, distortion = vq.kmeans(sift_des, k, iter=iters, thresh=threshold)

    print('K-Means for {0} clusters completed. Distortion is {1:1.6f}'.format(k, distortion))

    np.save('./dictionaries/k_{}.npy'.format(k), codebook)




