import numpy as np
import cv2 as cv
from keypoint_matching import compute_matches
from RANSAC import *



def demo():
    '''
    Demo code
    '''

    # specify paths
    path1 = './right.jpg'
    path2 = './left.jpg'

    # load images
    img1 = cv.imread(path1)
    img2 = cv.imread(path2)
    gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    # num iterations for RANSAC
    N = 100

    # num points to sample for RANSAC
    num_points = 10

    # compute matchings: query=boat1, target=boat2
    kp_query, des1, kp_train, des2, matches = compute_matches(gray1, gray2, threshold=0.75)

    # if we compute as compute_matches(img1, img2), then
    # query is kp1
    # train is kp2
    
    # extract all coordinates and put them into numpy arrays
    coords_query, coords_train = get_coords(kp_query, kp_train, matches)

    # perform RANSAC on the matches
    trafo_params, inliers, best_kp_idcs = RANSAC(N, coords_query, coords_train, num_points)
    print('Optimal parameters are {}, with {} inliers.'.format(trafo_params, inliers))

    # find the corners of the transformed query (right) image
    # cv2.warpAffine takes a 2x3 matrix, where the last column is the bias
    M = np.array([[trafo_params[0], trafo_params[1], trafo_params[4]], [trafo_params[2], trafo_params[3], trafo_params[5]]])
    trafo_query = cv.warpAffine(img1, M, (img1.shape[1],img1.shape[0])) # this gives a cropped image, if you make the out_dim twice as big it fits in, but there might be a better way..
    
    #cv.imshow('Right warped',trafo_query)
    #cv.waitKey(0)
    cv.imwrite('warped_right.jpg', trafo_query)
    
    
    

if __name__ == "__main__":
    
    demo()