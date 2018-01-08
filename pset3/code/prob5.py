## Default modules imported. Import more if you need to.

import numpy as np
from scipy.ndimage.interpolation import shift


#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################





## Fill out these functions yourself

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
def census(img):

    W = img.shape[1]
    H = img.shape[0]

    c = np.zeros((H,W), dtype='uint32')
    bit_pos=0
    for deltax in range(-2,3):
        for deltay in range(-2,3):
            if deltax==0 and deltay==0:
                continue
            #compare x with shifted copy (fill value of inf ensures out-of-bounds comparisons will be 0)
            bit = img>shift(img, (deltax, deltay), cval=float('inf'))
            #now shift this bit to an appropriate position in the representation
            #and add it to the census code
            c = c+(bit<<bit_pos)
            bit_pos += 1
    return c


# Given left and right image and max disparity D_max, return a disparity map
# based on matching with  hamming distance of census codes. Use the census function
# you wrote above.
#
# d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
# should be between 0 and D_max (both inclusive).
def smatch(left,right,dmax):
    #get x and y coordinates for the left image
    y, x = np.indices(left.shape)
    drange = np.tile(np.arange(dmax+1),np.size(x)).reshape(x.shape+(dmax+1,))
    #limit drange to values that are less than the x-values
    drange = np.minimum(drange, x[:,:,np.newaxis])
    #for each point, find the hamming distances associated with each d-value
    left_census = census(left)
    hamrange = np.empty(drange.shape)
    for i in range(drange.shape[-1]):
        right_image = right[y, x-drange[:,:,i]]
        right_census = census(right_image)
        hamrange[:,:,i] = hamdist(left_census, right_census)
    #for each point, find axis-2 index of minimum hamming value
    min_idx = np.argmin(hamrange, axis=2)
    #finally, use these indices to lookup the corresponding d-values
    d = drange[y,x,min_idx]
    #for each point, generate range of possible d values
    drange = np.tile(np.arange(dmax+1),np.size(x)).reshape((dmax+1,)+x.shape)
    #limit drange to values that are less than the x-values

    print(d)

    return d



########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = imread(fn('inputs/left.jpg'))
right = imread(fn('inputs/right.jpg'))

d = smatch(left,right,40)
census(left)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob5.png'),dimg)
