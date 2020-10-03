import numpy as np
import cv2 as cv

def compute_matches(img1, img2, threshold=0.75):
    '''
    Computes keypoints, discriptors and matches between two GRAYSCALE images
    Input:
        img1 - image 1
        img2 - image 2
        threshold - float, threshold for the ratio test
    Output:
        kp1 - list of KeyPoints
        kp2 - dito
        des1 - list(?) of descriptors
        des2 - dito
        matches - list of matches between kp1 and kp2, based on des1 an des2
    '''

    # perform sift to get keypoints
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # instantiate matcher object
    matcher = cv.BFMatcher_create()
    # get 2 best matches
    matches = matcher.knnMatch(des1, des2, k=2)
    # apply ratio test -> reject keypoints for which the second best
    # match is not much worse than the best match, as explained by D.Lowe
    # in his paper
    good_matches = []
    for i,j in matches:
        if i.distance < threshold * j.distance:
            good_matches.append([i])

    return kp1, des1, kp2, des2, good_matches

if __name__ == "__main__":

    # specify paths
    boat1 = './boat1.pgm'
    boat2 = './boat2.pgm'

    # load images
    img1 = cv.imread(boat1)
    img2 = cv.imread(boat2)
    gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    # compute matchings
    kp1, des1, kp2, des2, matches = compute_matches(gray1, gray2, threshold=0.75)

    # plot 10 random matches
    idcs = np.random.choice(np.arange(len(matches)), 10, replace=False)
    random_matches = np.array(matches)[idcs]
    match_img = cv.drawMatchesKnn(img1, kp1, img2, kp2, random_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('boat_matches.jpg', match_img)
