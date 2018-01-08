## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Different thresholds to try
T0 = 0.5
T1 = 1.0
T2 = 1.5


########### Fill in the functions below

# Return magnitude, theta of gradients of X
def grads(X):
    #define sobel kernels
    Sx = np.array([(-1,0,1), (-2,0,2), (-1,0,1)])
    Sy = -Sx.transpose()
    Gx = conv2(X, Sx, 'same')
    Gy = conv2(X, Sy, 'same')

    #compute magnitude
    H = np.sqrt(Gx**2 + Gy**2)

    #compute direction
    theta = np.arctan(Gy / Gx)

    return H,theta

def nms(E,H,theta):
    #round off theta to either horizontal, vertical, diag1 or diag2
    #so, to the nearest pi/4
    rows, cols = E.shape
    directions = np.around(theta/(np.pi/4)) #should be -2, -1, 0, 1, or 2
    edges_x, edges_y = np.where(E>0)
    filtered = np.zeros(E.shape)
    for (i,j) in zip(edges_x, edges_y):
        d = directions[i][j]
        h = H[i][j]
        if (d==-2 or d==2): #north-south direction
            if i == 0:
                filtered[i][j] = h > H[i+1][j]
            elif i == rows-1:
                filtered[i][j] = h > H[i-1][j]
            else:
                filtered[i][j] = h > H[i-1][j] and h > H[i+1][j]
        elif d == -1: #northwest-southeast direction
            if (i==0 and j==cols-1) or (i==rows-1 and j==cols-1):
                filtered[i][j] = 1
            elif (i==0 or j==0):
                filtered[i][j] = h > H[i+1][j+1]
            elif (i==rows-1 or j==cols-1):
                filtered[i][j] = h > H[i-1][j-1]
            else:
                filtered[i][j] = h > H[i-1][j-1] and h > H[i+1][j+1]
        elif d == 0: #east-west direction
            if j==0:
                filtered[i][j] = h > H[i][j+1]
            elif j==cols-1:
                filtered[i][j] = h > H[i][j-1]
            else:
                filtered[i][j] = h > H[i][j+1] and h > H[i][j-1]
        elif d == 1: #northeast-southwest direction
            if (i==0 and j==0) or (i==rows-1 and j==rows-1):
                filtered[i][j] = 1
            elif i==rows-1 or j==0:
                filtered[i][j] = h > H[i-1][j+1]
            elif i==0 or j==cols-1:
                filtered[i][j] = h > H[i+1][j-1]
            else:
                filtered[i][j] = h > H[i+1][j-1] and h > H[i-1][j+1]
    return filtered

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p3_inp.png')))/255.

H,theta = grads(img)

imsave(fn('outputs/prob3_a.png'),H/np.max(H[:]))

## Part b

E0 = np.float32(H > T0)
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/prob3_b_0.png'),E0)
imsave(fn('outputs/prob3_b_1.png'),E1)
imsave(fn('outputs/prob3_b_2.png'),E2)

E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/prob3_b_nms0.png'),E0n)
imsave(fn('outputs/prob3_b_nms1.png'),E1n)
imsave(fn('outputs/prob3_b_nms2.png'),E2n)
