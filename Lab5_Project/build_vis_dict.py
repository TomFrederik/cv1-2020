import numpy as np
import cv2 as cv
import os
import argparse
import scipy.cluster.vq as vq


class_names = ['airplane', 'bird', 'ship', 'horse', 'car']
class_nbrs = [1,2,3,4,5]


parser = argparse.ArgumentParser(description='Create visual dictionary.')
parser.add_argument('method', type=str, help='sift, surf or hog')
parser.add_argument('--color', type=str, help='rgb or gray', default='rgb')
parser.add_argument('--type', type=str, help='train or test', default='train')
args = parser.parse_args()

DATA_DIR = './data/{}/'.format(args.type)
class_dirs = [DATA_DIR + str(nbr) + '/' for nbr in class_nbrs]

if args.method == "sift":
    method = cv.xfeatures2d.SIFT_create()
elif args.method == "surf":
    method = cv.xfeatures2d.SURF_create()
elif args.method == "hog":
    method = cv.HOGDescriptor()

descriptors = []

for i, folder in enumerate(class_dirs):
    print("Parsing", folder)
    # get images of each class
    (_, _, files) = next(os.walk(folder))

    files = [folder + f for f in files][:50]

    # get Keypoint SIFT descriptors for each image
    for cur_file in files:
        img = cv.imread(cur_file)
        # TODO: implement RGB-SIFT
        # According to Canvas, we need to run SIFT on the three channels
        # and concatenate the 3 x 128 features of the descriptors
        if args.method == "hog":
            # TODO: this throws a weird error, no idea how OpenCVs HOG works
            # Canvas suggestion: use skimage instead
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            des = method.compute(img)
        else:
            _, des = method.detectAndCompute(img, None)

        for j in range(des.shape[0]):
            descriptors.append(des[j, :])


descriptors = np.array(descriptors)
print('Descriptors shape:', descriptors.shape)

# now perform K-means clustering on these descriptors
num_clusters = [400, 1000, 4000]
iters = 20
threshold = 1e-5

for k in num_clusters:
    codebook, distortion = vq.kmeans(descriptors, k, iter=iters, thresh=threshold)

    print('K-Means for {0} clusters completed. Distortion is {1:1.6f}'.format(k, distortion))

    target_dir = './dictionaries/{}_{}_{}/'.format(args.method, args.color, args.type)
    os.makedirs(target_dir, exist_ok=True)
    np.save(target_dir + 'k_{}.npy'.format(args.method, args.color, args.type, k), codebook)
