import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from harris_corner_detector import harris_corner_detection
import numpy as np
from PIL import Image

######################
# set parameters here:
######################
# thresholds are different for each image
threshold = 5000
# size of the window to look for local maxima, must be odd
window_size = 13
# sigma for gaussian smoothing
sigma_smooth = 3
# sigma used for the derivatives (using derivatives of Gaussians)
sigma_derivative = 1
#######################


fig = plt.figure()

folder = "person_toy"
filesfolder = os.listdir(folder)
filesfolder.sort()

images = []

for filename in filesfolder:
    #print(filename)
    img = cv2.imread(os.path.join(folder,filename))
    #convert BGR to RGB
    img = img[:, :, ::-1]
    img = plt.imshow(img, animated=True)
    imggray = np.asarray(Image.open(os.path.join(folder,filename)).convert("L")).astype(float)

    H, rows, cols = harris_corner_detection(imggray, threshold, window_size, sigma_smooth, sigma_derivative)
    #print("Number of corners:", len(rows))
    plt.scatter(cols, rows, s=16, animated=True)
    if img is not None:
        images.append([img])


ani = animation.ArtistAnimation(fig, images, interval=20, blit=True,
                                repeat_delay=1000)

########################################################
x = np.linspace(0,15,len(filesfolder))

#fig = plt.figure()
p1 = fig.add_subplot(111)

# set up empty lines to be updates later on
l1, = p1.plot([],[],'b')
l2, = p1.plot([],[],'r')

def gen1():
    i = 0.5
    while(True):
        yield i
        i += 0.1

def gen2():
    j = 0
    while(True):
        yield j
        j += 1

def run1(c):
    y = c*x
    l1.set_data(x,y)

def run2(c):
    y = c*x
    l2.set_data(x,y)

ani1 = animation.FuncAnimation(fig,run1,gen1,interval=20)
ani2 = animation.FuncAnimation(fig,run2,gen2,interval=20)
########################################################

plt.show()

ani.save('myAnimation.gif', fps=30)
