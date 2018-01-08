## Default modules imported. Import more if you need to.

import numpy as np
from scipy.ndimage.interpolation import shift

# A note on the census and bfilt functions: in my old implementation, I used the
# shift function from scipy, which hypothetically has the same effect as using shifted
# array indices but is more computationally expensive. I switched to the version of this
# function given in the old problem set answers to save time. However, I notice that
# this version of the function produces different-looking images, particularly
# for problem 3b. I included my old version of the census and bfilt functions
# for comparison.


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


# Given left and right grayscale images and max disparity D_max, build a HxWx(D_max+1) array
# corresponding to the cost volume. For disparity d where x-d < 0, fill a cost
# value of 24 (the maximum possible hamming distance).
#
# You can call the hamdist function above, and copy your census function from the
# previous problem set.

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
# def census(img):
#
#     W = img.shape[1]
#     H = img.shape[0]
#
#     c = np.zeros((H,W), dtype='uint32')
#     bit_pos=0
#     for deltax in range(-2,3):
#         for deltay in range(-2,3):
#             if deltax==0 and deltay==0:
#                 continue
#             #compare x with shifted copy (fill value of inf ensures out-of-bounds comparisons will be 0)
#             bit = img>shift(img, (deltax, deltay), cval=float('inf'))
#             #now shift this bit to an appropriate position in the representation
#             #and add it to the census code
#             c = c+(bit<<bit_pos)
#             bit_pos += 1
#     return c

def census(img):
    W = img.shape[1]; H = img.shape[0]
    c = np.zeros((H,W), dtype=np.uint32)

    inc = np.uint32(1)
    for dx in range(-2,3):
        for dy in range(-2,3):
            if dx==0 and dy==0:
                continue

            cx0 = np.maximum(0,-dx); dx0 = np.maximum(0,dx)
            cx1 = W-dx0; dx1 = W-cx0
            cy0 = np.maximum(0,-dy); dy0 = np.maximum(0,dy)
            cy1 = H-dy0; dy1 = H-cy0

            c[cy0:cy1, cx0:cx1] = c[cy0:cy1,cx0:cx1] + \
                (img[cy0:cy1,cx0:cx1] > img[dy0:dy1,dx0:dx1]) * inc
            inc *= 2

    return c


def buildcv(left,right,dmax):
    #note: I happened to do this anyway in the last problem set, so I'm reusing that code
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
    return hamrange


# Fill this out
# CV is the cost-volume to be filtered.
# X is the left color image that will serve as guidance.
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
#
# Feel free to look at the solution key for bfilt function form problem set 1.
# def bfilt(cv,X,K,sgm_s,sgm_i):
#     C = np.zeros(cv.shape)
#     norm_sums = np.zeros(cv.shape)
#
#     for k in np.ndindex(2*K+1, 2*K+1):
#         s = (K-k[0], K-k[1], 0)
#         Xshift = shift(X, s)
#         Xdiffs = ((Xshift - X)**2)/(2*sgm_i**2)
#         #in this case we want the sum squared difference for all 3 colors...I think
#         Xdiffs = np.sum(Xdiffs, axis=2)
#
#         ndiff = ((K-k[0])**2 + (K-k[1])**2)/(2*sgm_s**2)
#         ndiffs = np.full(cv.shape, ndiff)
#
#         B_k = np.exp(-ndiffs - Xdiffs[:,:,np.newaxis])
#         norm_sums += B_k
#         Cshift = shift(cv,s)
#         C += Cshift*B_k
#
#     C = C/norm_sums
#
#     return C

def bfilt(cv,X,K,sgm_s,sgm_i):
    H=X.shape[0]; W=X.shape[1]
    C = np.zeros(cv.shape)
    norm_sums = np.zeros(cv.shape)

    for x,y in np.ndindex(2*K+1, 2*K+1):
        x -= K; y -= K;

        #compute indices for shifted copies of X, cv
        if y<0:
            y1a = 0; y1b = -y; y2a = H+y; y2b = H
        else:
            y1a = y; y1b = 0; y2a = H; y2b = H-y

        if x<0:
            x1a=0; x1b=-x; x2a=W+x; x2b = W
        else:
            x1a=x; x1b=0; x2a=W; x2b=W-x

        #compute X-based term
        xdiffs = X[y1a:y2a, x1a:x2a,:] - X[y1b:y2b, x1b:x2b]
        xdiffs = np.sum(xdiffs**2, axis=2, keepdims=False)
        xdiffs = xdiffs/(2*sgm_i**2)

        #compute n-based term
        ndiffs = np.float32(x**2 + y**2)/(2*sgm_s**2)

        #compute the B matrix for this k
        B_k = np.exp(-ndiffs - xdiffs)[:,:,np.newaxis]
        norm_sums[y1b:y2b, x1b:x2b] += B_k
        C[y1b:y2b,x1b:x2b] += B_k*cv[y1a:y2a, x1a:x2a, :]

    return C/norm_sums



########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)

cv0 = buildcv(left_g,right_g,50)

cv1 = bfilt(cv0,left,5,2,0.5)


d0 = np.argmin(cv0,axis=2)
d1 = np.argmin(cv1,axis=2)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d0.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d0.shape[0],d0.shape[1],3])
imsave(fn('outputs/prob2a.jpg'),dimg)

dimg = cm.jet(np.minimum(1,np.float32(d1.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d1.shape[0],d1.shape[1],3])
imsave(fn('outputs/prob2b.jpg'),dimg)
