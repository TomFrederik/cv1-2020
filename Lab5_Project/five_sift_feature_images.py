import numpy as np
import cv2 as cv


class_names = ['airplane', 'bird', 'ship', 'horse', 'car']
class_nbrs = [1,2,3,4,5]

DATA_DIR = './data/train/'

img1 = DATA_DIR + '1/10.png'
img2 = DATA_DIR + '2/0.png'
img3 = DATA_DIR + '3/9.png'
img4 = DATA_DIR + '4/3.png'
img5 = DATA_DIR + '5/18.png'

images = [img1, img2, img3, img4, img5]

# perform sift to get keypoints
sift = cv.xfeatures2d.SIFT_create()

for i, img in enumerate(images):
    color = cv.imread(img)

    kp, des = sift.detectAndCompute(color, None)
    new_img = cv.drawKeypoints(color,kp,color,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imwrite('./sift_examples/'+class_names[i]+'.jpg', new_img)