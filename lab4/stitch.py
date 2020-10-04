import numpy as np
import cv2 as cv
from keypoint_matching import compute_matches
from RANSAC import *
import matplotlib.pyplot as plt


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
    gray1= cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2= cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

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

    a, b, _ = img1.shape
    corners = np.array([[0, 0, b, b], [0, a, 0, a]])
    warped_corners = M[:, :2] @ corners + M[:, 2, None]

    bottom = np.floor(np.max(warped_corners[1])).astype(int)
    right = np.floor(np.max(warped_corners[0])).astype(int)

    trafo_query = cv.warpAffine(img1, M, (right, bottom))

    cv.imwrite('warped_right.jpg', trafo_query)

    bottom = max(bottom, img2.shape[0])
    right = max(right, img2.shape[1])

    # How do we find the new size?
    # This takes the smallest rectangle that contains everything
    new_size = (bottom, right, 3)

    stitched_img = np.zeros(new_size)
    stitched_img[:trafo_query.shape[0], :trafo_query.shape[1], :] = trafo_query
    stitched_img[:img2.shape[0], :img2.shape[1], :] = img2
    stitched_img /= 255
    # convert BGR to RGB
    stitched_img = stitched_img[:, :, ::-1]
    plt.imshow(stitched_img)
    plt.show()


if __name__ == "__main__":
    demo()
