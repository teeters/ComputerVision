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


# Do SGM. First compute the augmented / smoothed cost volumes along 4
# directions (LR, RL, UD, DU), and then compute the disparity map as
# the argmin of the sum of these cost volumes.
def SGM(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]

    #lr pass: compute Cbar and Z for each X (at each pixel)
    #Cbar and Z will have dimensions (H,W,D)
    lrCbar = np.zeros(cv.shape)
    rlCbar = np.zeros(cv.shape)
    lrCbar[:,0,:] = cv[:,0,:]
    rlCbar[:,-1,:] = cv[:,-1,:]
    for x in range(1, W-1):
        #fx = x going forward, rx = x going backward
        fx = x
        rx = W-1-x
        #remember cv is indexed [y,x,d]
        #compute C^tilde[x-1, dp] (has dimensions (H,D))
        fC_t = lrCbar[:,fx-1,:] - np.amin(lrCbar[:,fx-1,:], axis=1)[:,np.newaxis]
        rC_t = rlCbar[:,fx+1,:] - np.amin(rlCbar[:,fx+1,:], axis=1)[:,np.newaxis]
        #compute min(C_t[x-1, dp] + S(d,dp)), which we know to be one of four values
        fv1 = rv1 = P2 #values for dp = d
        fv2 = np.roll(fC_t, 1, axis=1)+P1 #values for dp = d-1
        rv2 = np.roll(rC_t, 1, axis=1)+P1
        fv2[:,-1] = 1000 #prevent argmin from choosing d-1 for d=0
        rv2[:,-1] = 1000
        fv3 = np.roll(fC_t, -1, axis=1)+P1 #values for dp = d+1
        rv3 = np.roll(rC_t, -1, axis=1)+P1
        fv3[:,0] = 1000
        rv3[:,0] = 1000
        fv4 = fC_t #values for dp = d
        rv4 = rC_t
        lr_forward_min = np.minimum(np.minimum(fv1, fv2), np.minimum(fv3, fv4))
        rl_forward_min = np.minimum(np.minimum(rv1, rv2), np.minimum(rv3, rv4))
        lrCbar[:,fx,:] = cv[:,fx,:] + lr_forward_min
        rlCbar[:,rx,:] = cv[:,rx,:] + rl_forward_min

    #vertical pass.
    Cbar_up = np.zeros(cv.shape)
    Cbar_down = np.zeros(cv.shape)
    Cbar_up[0,:,:] = cv[0,:,:]
    Cbar_down[-1,:,:] = cv[-1,:,:]
    for y in range(1, H-1):
        y_up = y
        y_down = H-1-y
        #compute C_tilde (dimensions W,D)
        Ct_up = Cbar_up[y_up-1,:,:] - np.amin(Cbar_up[y_up-1,:,:], axis=1)[:,np.newaxis]
        Ct_down = Cbar_down[y_down+1,:,:] - np.amin(Cbar_down[y_down+1,:,:], axis=1)[:,np.newaxis]
        #compute min(Ct[y-1, dp] + S(d,dp))
        v1_up = v1_down = P2
        v2_up = np.roll(Ct_up, 1, axis=1)+P1
        v2_down = np.roll(Ct_down, 1, axis=1)+P1
        v2_up[:,-1] = v2_down[:,-1] = 1000
        v3_up = np.roll(Ct_up, -1, axis=1)+P1
        v3_down = np.roll(Ct_down, -1, axis=1) + P1
        v3_up[:,0] = v3_down[:,0] = 1000
        v4_up = Ct_up
        v4_down = Ct_down
        forward_min_up = np.minimum(np.minimum(v1_up, v2_up), np.minimum(v3_up, v4_up))
        forward_min_down = np.minimum(np.minimum(v1_down, v2_down), np.minimum(v3_down, v4_down))
        Cbar_up[y_up,:,:] = cv[y_up,:,:] + forward_min_up
        Cbar_down[y_down,:,:] = cv[y_down,:,:] + forward_min_down

    #sum cost functions and minimize
    Cbar = Cbar_up + Cbar_down + lrCbar + rlCbar
    d = np.argmin(Cbar, axis=2)

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
d = SGM(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3b.jpg'),dimg)
