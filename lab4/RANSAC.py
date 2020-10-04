import numpy as np
import cv2 as cv
from keypoint_matching import compute_matches
from matplotlib import pyplot as plt
import math

def get_coords(query, train, matches):
    '''
    Extract the match coordinates from query and train and puts them into numpy arrays
    '''
    coords_query = np.zeros((len(matches), 2))
    coords_train = np.zeros((len(matches), 2))

    for i in range(len(matches)):
        coords_query[i] = np.array(query[matches[i][0].queryIdx].pt)
        coords_train[i] = np.array(train[matches[i][0].trainIdx].pt)

    return coords_query, coords_train

def RANSAC(num_iters, coords_query, coords_train, num_points=10):
    '''
    Performs the RANSAC algorithm to find the affine transformation between
    two images.
    Input:
        num_iters - number of iterations
        coords_query - coordinates of matched keypoints in query, shape is (num_matches, 2)
        coords_train - coordinates of matched keypoints in target, shape is (num_matches, 2)
        num_points - int, number of points to find the LS solution
    Output:
        best_params - best found parameters for the affine trafo
        max_inliers - number of inliers for the best trafo
        best_kp_idcs - ndarray, set of indices keypoints which led to the best fit
    '''
    assert num_points >= 6, 'Expected num_points >= 6, to find a LS solution, but got {}'.format(num_points)
    assert coords_query.shape == coords_train.shape, 'Coords have different shapes: {} and {}'.format(coords_query.shape, coords_train.shape)

    best_params = np.zeros(6)
    max_inliers = 0
    best_kp_idcs = []

    for _ in range(num_iters):
        # sample points
        idcs = np.random.choice(np.arange(coords_query.shape[0]), num_points, replace=False)

        # construct matrix A and vector b
        A = np.zeros((2*num_points,6))
        b = np.zeros((2*num_points))

        for i, idx in enumerate(idcs):
            b[2*i:2*i+2] = np.array(coords_train[idx])
            A[2*i:2*i+2,:] = np.array([[*coords_query[idx],0,0,1,0],\
                                   [0,0,*coords_query[idx],0,1]])


        # find least squares solution to Ax=b, via pseudoinverse
        A_dagger = np.linalg.pinv(A)
        x = A_dagger @ b

        # use current params to apply the transformation to all matches
        B = np.array([[x[0],x[1]], [x[2], x[3]]]) # trafo matrix
        bias = np.array([x[4], x[5]])

        trafo_out = (coords_query @ B.T) + bias[None,:]

        # compute number of inliers
        abs_diff = np.sqrt(np.sum((trafo_out - coords_train)**2, axis=1))
        num_inliers = np.shape(abs_diff[abs_diff<=10])[0]

        # update trafo parameters if this is the best one so far
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_params = x
            best_kp_idcs = np.array(idcs, dtype=np.int32)

    return best_params, max_inliers, best_kp_idcs


def warp(img_orig, trafo_params):
    '''
    Applies an affine transformation for given parameters on a grid of coordinates
    and returns the transformed coordinates, rounded to the next integer
    '''
    coords_orig = np.indices((img_orig.shape[0],img_orig.shape[1])).T
    coords_orig = coords_orig.reshape(-1,2)

    B = np.array([[trafo_params[0],trafo_params[1]], [trafo_params[2], trafo_params[3]]]) # trafo matritrafo_params
    bias = np.array([trafo_params[4], trafo_params[5]])

    transf_coord = np.array(np.around((coords_orig @ np.linalg.inv(B).T) - bias[None,:]), dtype=np.int32)

    #find minimum and maximum x and y values to be able to fit the warped image into a picture frame
    min_x, max_x, min_y, max_y = math.inf, -math.inf, math.inf, -math.inf
    for i in range(len(transf_coord)):
        if transf_coord[i][0] < min_x:
            min_x = transf_coord[i][0]
        if transf_coord[i][0] > max_x:
            max_x = transf_coord[i][0]
        if transf_coord[i][1] < min_y:
            min_y = transf_coord[i][1]
        if transf_coord[i][1] > max_y:
            max_y = transf_coord[i][1]

    #offset the coords to have min_x value of 0 and min_y value of 0
    for j in range(len(transf_coord)):
        transf_coord[j][0] = transf_coord[j][0] - min_x
        transf_coord[j][1] = transf_coord[j][1] - min_y

    new_im = np.zeros((max_x-min_x+1,max_y-min_y+1,img_orig.shape[2]))
    new_im[transf_coord[:,0], transf_coord[:,1], :] = img_orig[coords_orig[:,0], coords_orig[:,1],:]

    return new_im.astype(int)

def demo():
    '''
    runs the script
    '''

    # specify paths
    boat1 = './boat1.pgm'
    boat2 = './boat2.pgm'

    # load images
    img1 = cv.imread(boat1)
    img2 = cv.imread(boat2)
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

    # create plot of two images next to each other with lines from keypoints in img1 to their mapped coords in img2
    # compute new coords of query
    # use current params to apply the transformation to the best keypoints
    # round the output to the neares integers
    B = np.array([[trafo_params[0],trafo_params[1]], [trafo_params[2], trafo_params[3]]]) # trafo matritrafo_params
    bias = np.array([trafo_params[4], trafo_params[5]])
    trafo_out = np.array(np.around((coords_query[best_kp_idcs] @ B.T) + bias[None,:]), dtype=np.int32)

    # round best_kp to nearest integers
    best_kp = np.array(np.around(coords_query[best_kp_idcs]), dtype=np.int32)

    # stack images to plot next to each other
    hstack_image = np.hstack((img1, img2))
    #print(hstack_image.shape) #, (680,1700,3)

    # compute offset for plotting the images next to each other
    offset = np.array([int(img1.shape[1]),0], dtype=np.int32)
    # draw red lines between the best keypoints
    for i in range(best_kp.shape[0]):
        pt1 = tuple(best_kp[i])
        pt2 = tuple(trafo_out[i] + offset)
        cv.line(hstack_image, pt1, pt2, (0,0,255), 3)

    cv.imwrite('./side_by_side.jpg',hstack_image)

    ##use transformation parameters of img1 -> img2 to construct warped images

    #own function
    new_img1 = warp(img1,trafo_params)

    #cv2 warpAffine
    M = np.array([[trafo_params[0], trafo_params[1], trafo_params[4]],
                    [trafo_params[2], trafo_params[3], trafo_params[5]]])
    a, b, _ = img1.shape
    corners = np.array([[0, 0, b, b], [0, a, 0, a]])
    warped_corners = M[:, :2] @ corners + M[:, 2, None]
    bottom = np.floor(np.max(warped_corners[1])).astype(int)
    right = np.floor(np.max(warped_corners[0])).astype(int)
    top = np.floor(np.min(warped_corners[1])).astype(int)
    left = np.floor(np.min(warped_corners[0])).astype(int)
    M[0,2] = M[0,2] - left
    M[1,2] = M[1,2] - top
    new_img1_cv = cv.warpAffine(img1,M,(right-left,bottom-top))

    ##use transformation parameters of img2 -> img1 to construct warped images

    #compute parameters for img2 to img1
    kp_query, _, kp_train, _, matches = compute_matches(gray2, gray1, threshold=0.75)
    coords_query, coords_train = get_coords(kp_query, kp_train, matches)
    trafo_params2, _, _ = RANSAC(N, coords_query, coords_train, num_points)

    #own function
    new_img2 = warp(img2,trafo_params2)

    #cv2 warpAffine
    M = np.array([[trafo_params2[0], trafo_params2[1], trafo_params2[4]],
                    [trafo_params2[2], trafo_params2[3], trafo_params2[5]]])
    a, b, _ = img2.shape
    corners = np.array([[0, 0, b, b], [0, a, 0, a]])
    warped_corners = M[:, :2] @ corners + M[:, 2, None]
    bottom = np.floor(np.max(warped_corners[1])).astype(int)
    right = np.floor(np.max(warped_corners[0])).astype(int)
    top = np.floor(np.min(warped_corners[1])).astype(int)
    left = np.floor(np.min(warped_corners[0])).astype(int)
    M[0,2] = M[0,2] - left
    M[1,2] = M[1,2] - top
    new_img2_cv = cv.warpAffine(img2, M, (right - left, bottom - top))

    #visualize all warped images
    fig=plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(new_img1,vmin=0,vmax=255)
    plt.axis('off')
    plt.title('Own function (img1 -> img2)')
    plt.subplot(2, 2, 2)
    plt.imshow(new_img1_cv,vmin=0,vmax=255)
    plt.axis('off')
    plt.title('warpAffine (img1 -> img2)')
    plt.subplot(2, 2, 3)
    plt.imshow(new_img2,vmin=0,vmax=255)
    plt.axis('off')
    plt.title('Own function (img2 -> img1)')
    plt.subplot(2, 2, 4)
    plt.imshow(new_img2_cv,vmin=0,vmax=255)
    plt.axis('off')
    plt.title('warpAffine (img2 -> img1)')
    plt.show()

if __name__ == "__main__":
    demo()
