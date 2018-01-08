## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.ndimage.interpolation import shift

# Fill this out
# X is input color image
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
def bfilt(X,K,sgm_s,sgm_i):
    Y = np.zeros(X.shape)
    norm_sums = np.zeros(X.shape)

    for k in np.ndindex(2*K+1, 2*K+1):
        s = (K-k[0], K-k[1], 0)
        Xshift = shift(X, s)
        Xdiffs = ((Xshift - X)**2)/(2*sgm_i**2)

        ndiff = ((K-k[0])**2 + (K-k[1])**2)/(2*sgm_s**2)
        ndiffs = np.full(X.shape, ndiff)

        B_k = np.exp(-ndiffs - Xdiffs)
        norm_sums += B_k
        Y += Xshift*B_k

    Y = Y/norm_sums
    return Y


########################## Support code below

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img1 = np.float32(imread(fn('inputs/p4_nz1.png')))/255.
img2 = np.float32(imread(fn('inputs/p4_nz2.png')))/255.

K=9

print("Creating outputs/prob4_1_a.png")
im1A = bfilt(img1,K,2,0.5)
imsave(fn('outputs/prob4_1_a.png'),clip(im1A))


print("Creating outputs/prob4_1_b.png")
im1B = bfilt(img1,K,4,0.25)
imsave(fn('outputs/prob4_1_b.png'),clip(im1B))

print("Creating outputs/prob4_1_c.png")
im1C = bfilt(img1,K,16,0.125)
imsave(fn('outputs/prob4_1_c.png'),clip(im1C))

# Repeated application
print("Creating outputs/prob4_1_rep.png")
im1D = bfilt(img1,K,2,0.125)
for i in range(8):
    im1D = bfilt(im1D,K,2,0.125)
imsave(fn('outputs/prob4_1_rep.png'),clip(im1D))

# Try this on image with more noise
print("Creating outputs/prob4_2_rep.png")
im2D = bfilt(img2,2,8,0.125)
for i in range(16):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/prob4_2_rep.png'),clip(im2D))
