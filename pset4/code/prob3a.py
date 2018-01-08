## Default modules imported. Import more if you need to.

import numpy as np

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


# Implement the forward-backward viterbi method to smooth
# only along horizontal lines. Assume smoothness cost of
# 0 if disparities equal, P1 if disparity difference <= 1, P2 otherwise.
#
# Function takes in cost volume cv, and values of P1 and P2
# Return the disparity map
def viterbilr(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]

    #for a given X, S will be a (Y,d,dp) array containing 0 where d=dp, P1 where |d-dp|=1, P2 otherwise
    S_id = np.indices((H,D,D))
    id1 = np.where(S_id[1,:,:,:] == S_id[2,:,:,:]) #first condition
    id2 = np.where(np.abs(S_id[1,:,:,:]-S_id[2,:,:,:]) == 1) #second condition
    S = np.full((H,D,D), P2)
    S[id1] = 0
    S[id2] = P1

    #forward pass: compute Cbar and Z for each X (at each pixel)
    #Cbar and Z will have dimensions (H,W,D)
    Cbar = np.zeros(cv.shape)
    Z = np.zeros(cv.shape, dtype=int)
    Cbar[:,0,:] = cv[:,0,:]
    for x in range(1, W-1):
        #remember cv is indexed [y,x,d]
        #compute C^tilde[x-1, dp] (has dimensions (H,D))
        C_t = Cbar[:,x-1,:] - np.amin(Cbar[:,x-1,:], axis=1)[:,np.newaxis]
        #compute min(C_t[x-1, dp] + S(d,dp)), which we know to be one of four values
        v1 = P2 #values for dp = d
        v2 = np.roll(C_t, 1, axis=1)+P1 #values for dp = d-1
        v2[:,-1] = 1000
        v3 = np.roll(C_t, -1, axis=1)+P1 #values for dp = d+1
        v3[:,0] = 1000
        v4 = C_t #values for dp = d
        forward_min = np.minimum(np.minimum(v1, v2), np.minimum(v3, v4))
        Cbar[:,x,:] = cv[:,x,:] + forward_min

        #compute z[x+1,d] = argmin over dp (S(d,dp) + Cbar[x,dp]
        Z[:,x,:] = np.argmin(S+Cbar[:,x-1,:,np.newaxis], axis=1)

    #backward pass:
    d = np.zeros((H,W), dtype=int)
    d[:,-1] = np.argmin(Cbar[:,-1,:], axis=1)
    for x in range(W-2, -1, -1):
        d[:,x] = Z[:,x+1,:][np.arange(H), d[:,x+1]]
    return d

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

cv = buildcv(left_g,right_g,50)
d = viterbilr(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3a.jpg'),dimg)
