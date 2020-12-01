import numpy as np
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt




# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 1
ncols = 2


# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)


sp = plt.subplot(1, 2, 0)
sp.axis('Off') # Don't show axes (or gridlines)

image_path = 'tmp/sample/Yaqo6z_1.jpg'
im = cv.imread(image_path)
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

img = mpimg.imread(img_path)
plt.imshow(img)