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



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid like in slides
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Just scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############# Main Program


img = np.float32(imread(fn('inputs/p6_inp.png')))/255.

# Visualize pyramids
pyr = im2wv(img,1)
imsave(fn('outputs/prob6a_1.png'),clip(vis(pyr)))

pyr = im2wv(img,2)
imsave(fn('outputs/prob6a_2.png'),clip(vis(pyr)))

pyr = im2wv(img,3)
imsave(fn('outputs/prob6a_3.png'),clip(vis(pyr)))

# Inverse transform to reconstruct image
im = clip(wv2im(pyr))
imsave(fn('outputs/prob6b.png'),im)

# Zero out some levels and reconstruct
for i in range(len(pyr)-1):

    for j in range(3):
        pyr[i][j][...] = 0.

    im = clip(wv2im(pyr))
    imsave(fn('outputs/prob6b_%d.png' % i),im)
