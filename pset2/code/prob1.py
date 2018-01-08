## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

def im2wv(img,nLev):
    V = []
    X = img
    for level in range(nLev):
        imax, jmax = X.shape
        A = X[0:imax:2, 0:jmax:2]
        B = X[0:imax:2, 1:jmax:2]
        C = X[1:imax:2, 0:jmax:2]
        D = X[1:imax:2, 1:jmax:2]

        L = 1/2 * (A+B+C+D)
        H1 = 1/2 * (B+D-A-C)
        H2 = 1/2 * (C+D-A-B)
        H3 = 1/2 * (A+D-B-C)

        V.append([H1, H2, H3])
        if level == nLev - 1:
            V.append(L)
        X = L

    return V

def wv2im(pyr):
    # d = 1/2 (L+H1+H2+H3)
    # a = L+H3-d
    # b = L+H1-d
    # c = L+H2-d
    L = pyr[-1]

    for lev in range(len(pyr)-2, -1, -1):
        H1, H2, H3 = pyr[lev]
        d = 1/2 * (H1 + H2 + H3 + L)
        a = L+H3 - d
        b = L+H1-d
        c = L+H2-d

        imax, jmax = tuple(2*n for n in a.shape)
        L = np.empty((imax, jmax))
        L[0:imax:2, 0:jmax:2] = a
        L[0:imax:2, 1:jmax:2] = b
        L[1:imax:2, 0:jmax:2] = c
        L[1:imax:2, 1:jmax:2] = d

    return L


# Fill this out
# You'll get a numpy array/image of coefficients y
# Return corresponding coefficients x (same shape/size)
# that minimizes (x - y)^2 + lmbda * abs(x)
def denoise_coeff(y,lmbda):
    x1 = y+.5*lmbda
    x2 = y-.5*lmbda
    x = np.empty(y.shape)
    #note: I can almost choose the min between x1 and x2 using a numpy array function,
    #but I can't find a way to pass a 'key' function for calculating the error value.
    for i in np.ndindex(y.shape):
        x[i] = min((x1[i], x2[i]), key=lambda a: (a-y[i])**2+lmbda*abs(a))
    return x



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program

lmain = 0.88

img = np.float32(imread(fn('inputs/p1.png')))/255.

pyr = im2wv(img,4)
for i in range(len(pyr)-1):
    for j in range(2):
        pyr[i][j] = denoise_coeff(pyr[i][j],lmain/(2**i))
    pyr[i][2] = denoise_coeff(pyr[i][2],np.sqrt(2)*lmain/(2**i))

im = wv2im(pyr)
imsave(fn('outputs/prob1.png'),clip(im))
