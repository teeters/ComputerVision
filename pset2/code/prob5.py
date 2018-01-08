## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d
from skimage.io import imread, imsave

def conv2(a, b):
    return convolve2d(a, b, 'same')

def scalardot(a,b):
    return np.sum(a*b)

## Fill out these functions yourself


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxWx3.
#
# Be careful about division by 0.
#
# Implement using conjugate gradient, with a weight = 0 for mask == 0, and proportional
# to n_z^2 elsewhere. See slides.

def ntod(nrm, mask, lmda):
    #Z = argmin Z^TQZ - 2Z^Tb + c
    Z = np.zeros(mask.shape)
    #derivative kernels
    fx = np.asarray([0.5,0,-0.5]).reshape((1,3))
    fy = np.asarray([-0.5,0,0.5]).reshape((3,1))
    fr = np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])/9
    fx_flip = -fx
    fy_flip = -fy
    fr_flip = np.flip(np.flip(fr, 0), 1)

    #derivatives
    gx = -nrm[:,:,0]/(nrm[:,:,2]+10**-12)
    gy = -nrm[:,:,1]/(nrm[:,:,2]+10**-12)

    #initialize variables
    k=0
    w = np.where(mask, nrm[:,:,2]**2,0)
    b = conv2(gx*w, fx_flip)+conv2(gy*w, fy_flip)
    r = b
    p = r

    #iteratively compute conjugate gradient
    while(k<200):
        Qp = conv2(conv2(p,fx)*w, fx_flip) + conv2(conv2(p,fy)*w, fy_flip) + \
            lmda*conv2(conv2(p,fr), fr_flip)
        alpha = scalardot(r, r) / scalardot(p, Qp)
        Z = Z + alpha * p
        print(np.sqrt(np.sum((alpha*p)**2)))
        new_r = r - alpha * Qp
        beta = scalardot(new_r, new_r)/scalardot(r,r)
        p = new_r + beta*p
        r = new_r
        k += 1
        #print(np.sqrt(np.sum(Z**2)))
    return Z


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
# nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-7)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
