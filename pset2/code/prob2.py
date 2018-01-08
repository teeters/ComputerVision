## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

## Take color image, and return 'white balanced' color image
## based on gray world, as described in Problem 2(a). For each
## channel, find the average intensity across all pixels.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2a(img):
    #calculate mean RGB values
    color_avg = np.mean(img,(0,1))
    #take inverse. Assumption: no zeros
    color_avg = 1.0/color_avg
    #normalize to sum of 3
    color_avg = color_avg/sum(color_avg)*3
    return img * color_avg


## Take color image, and return 'white balanced' color image
## based on description in Problem 2(b). In each channel, find
## top 10% of the brightest intensities, take their average.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2b(img):
    #get a vector of pixels sorted by intensity
    pixels = img.reshape((img.shape[0]*img.shape[1], 3))
    red_sorted = np.sort(pixels[:,0])
    green_sorted = np.sort(pixels[:,1])
    blue_sorted = np.sort(pixels[:,2])
    #for each color, find the top 10% of pixels and calculate the average
    toprange = int(img.shape[0]*img.shape[1]/10)
    red_avg = np.mean(red_sorted[-toprange:])
    green_avg = np.mean(green_sorted[-toprange:])
    blue_avg = np.mean(blue_sorted[-toprange:])

    return img / np.asarray([red_avg, green_avg, blue_avg])


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program
im1 = np.float32(imread(fn('inputs/CC/ex1.jpg')))/255.
im2 = np.float32(imread(fn('inputs/CC/ex2.jpg')))/255.
im3 = np.float32(imread(fn('inputs/CC/ex3.jpg')))/255.


im1a = balance2a(im1)
im2a = balance2a(im2)
im3a = balance2a(im3)

imsave(fn('outputs/prob2a_1.png'),clip(im1a))
imsave(fn('outputs/prob2a_2.png'),clip(im2a))
imsave(fn('outputs/prob2a_3.png'),clip(im3a))

im1b = balance2b(im1)
im2b = balance2b(im2)
im3b = balance2b(im3)

imsave(fn('outputs/prob2b_1.png'),clip(im1b))
imsave(fn('outputs/prob2b_2.png'),clip(im2b))
imsave(fn('outputs/prob2b_3.png'),clip(im3b))
