import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from harris_corner_detector import harris_corner_detection
import numpy as np
from PIL import Image
from lucas_kanade_copy import lucas_kanade
import copy
#
# ######################
# # set parameters here for Harris Corner Detector:
# ######################
# thresholds are different for each image
threshold = 5000
# size of the window to look for local maxima, must be odd
window_size = 13
# sigma for gaussian smoothing
sigma_smooth = 3
# sigma used for the derivatives (using derivatives of Gaussians)
sigma_derivative = 1
#######################

def tagCorners(image,rows,cols):
    """image is expected to be RGB. Red crosses will be placed on the coordinates given by rows and cols"""

    try:
        for i in range(7):
            rowsmin = rows-i*np.ones(len(rows)).astype(int)
            rowsplus = rows+i*np.ones(len(rows)).astype(int)
            colsmin = cols-i*np.ones(len(rows)).astype(int)
            colsplus = cols+i*np.ones(len(rows)).astype(int)
            image[rowsmin,cols,:] = [255,0,0]
            image[rowsplus,cols,:] = [255,0,0]
            image[rows,colsmin,:] = [255,0,0]
            image[rows,colsplus,:] = [255,0,0]
    except:
        print("Corner markers (red crosses) out of bound of the image")

    return image

def updateCorners(rows,cols,flows,image_size,window_size):
    for i in range(len(rows)):
        windowindex = findWindowIndex(rows[i],cols[i],window_size,image_size)
        rows[i] = rows[i] + flows[windowindex][0]
        cols[i] = cols[i] + flows[windowindex][1]

    return rows, cols

def findWindowIndex(row,col,window_size,image_size):
    windowcol = int(col/window_size)
    windowrow = int(row/window_size)
    totalrows = int(image_size[0]/window_size)
    windowindex = windowrow + totalrows*windowcol

    return windowindex

fig = plt.figure()

folder = "person_toy"
filesfolder = os.listdir(folder)
filesfolder.sort()

#import images and convert to expected type and expected colorchannels
imagesRGB = []
imagesgray = []
for filename in filesfolder:
    imgBGR = cv2.imread(os.path.join(folder,filename))
    imgRGB = imgBGR[:, :, ::-1]
    imggray = np.asarray(Image.open(os.path.join(folder,filename)).convert("L")).astype(float)
    imagesRGB.append(imgRGB)
    imagesgray.append(imggray)


imagesplt_persontoy = []

#create starting point: find corners in frame 1 and mark them
H, rows, cols = harris_corner_detection(imagesgray[0], threshold, window_size, sigma_smooth, sigma_derivative)
firstframe = tagCorners(imagesRGB[0],rows,cols)
imgplt = plt.imshow(firstframe)
imagesplt_persontoy.append([imgplt])

for i in range(len(imagesgray)-100):
    print(i)
    flows = lucas_kanade(imagesgray[i],imagesgray[i+1],window_size=100)
    rows, cols = updateCorners(rows,cols,flows,imagesgray[i].shape,window_size=100)
    print(rows)
    print(cols)
    imagesplt_persontoy.append([plt.imshow(tagCorners(imagesRGB[i+1],rows,cols))])



# imgplt = plt.imshow(imgRGB, animated=True)
# imagesplt_persontoy.append([imgplt])
print(len(imagesplt_persontoy))
ani_persontoy = animation.ArtistAnimation(fig, imagesplt_persontoy, interval=100, blit=True,
                                repeat_delay=100)
# ani_pingpong = animation.ArtistAnimation(fig, imagesplt_pingpong, interval=500, blit=True,
#                                 repeat_delay=1000)

plt.show()

#IF YOU WANT TO SAVE ANIMATIONS TO GIF, PLEASE UNCOMMENT CODE BELOW
ani_persontoy.save('persontoy.gif', fps=30)
#ani_pingpong.save('pingpong.gif', fps=30)
