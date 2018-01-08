## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
import math

#pset1 function for a same convolution matrix with circular padding in the Fourier domain.
def kernpad(K,size):
    h, w = size[0:2]
    kh, kw = K.shape[0:2]
    padding_h = int((h-kh) / 2)
    remainder_h = (h-kh)%2
    padding_w = int((w-kw) / 2)
    remainder_w = (w-kw)%2

    new_K = np.pad(K, [(padding_h+remainder_h, padding_h), (padding_w+remainder_w, padding_w)], mode='constant')
    rolldist = int(w*(padding_h+int(kh/2)+.5)) + w%2

    new_K = np.roll(new_K, rolldist)
    return new_K

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
# Implement in Fourier Domain / Frankot-Chellappa
def ntod(nrm, mask, lmda):
    #compute g matrices
    gx = -nrm[:,:,0]/(nrm[:,:,2]+10**-12)
    gy = -nrm[:,:,1]/(nrm[:,:,2]+10**-12)
    print(gx)
    print(gy)
    #take Fouriers
    Gx = np.fft.fft2(gx)
    Gy = np.fft.fft2(gy)

    #compute gradient kernels
    fx = np.asarray([0.5,0,-0.5]).reshape((1,3))
    fy = np.asarray([-0.5,0,0.5]).reshape((3,1))
    #pad and take Fouriers and conjugates
    Fx = np.fft.fft2(kernpad(fx, mask.shape))
    Fy = np.fft.fft2(kernpad(fy, mask.shape))
    Fx_conj = np.conj(Fx)
    Fy_conj = np.conj(Fy)

    #compute regularizer kernel and take Fourier
    fr = np.full((3,3), -1/9)
    fr[1,1] = 8/9
    Fr = np.fft.fft2(kernpad(fr, mask.shape))

    #compute Fourier of Z
    Fz = (Fx_conj*Gx + Fy_conj*Gy) / (np.abs(Fx)**2+np.abs(Fy)**2+lmda*np.abs(Fr)**2 + 10**(-12))
    Fz[0,0] = 0

    #invert transform to get the depth map Z
    Z = np.fft.ifft2(Fz)
    print(Z)

    return np.real(Z)


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
Z = ntod(nrm,mask,1e-6)


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

# #start by computing gx and gy, matrices of derivatives
# #gx = -nx/nz
# #gy = -ny/nz
# #both are zero where mask is zero
# nx = nrm[:,:,0]
# ny = nrm[:,:,1]
# nz = nrm[:,:,2]
# Gx = np.where(mask, -nx/nz, 0)
# Gx = np.fft.fft2(Gx)
# Gy = np.where(mask, -ny/nz, 0)
# Gy = np.fft.fft2(Gy)
#
# #compute Fr, the Fourier transform of the regularizer kernel
# r_kernel = np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])/9
# Fr = np.fft.fft2(kernpad(r_kernel, nrm.shape[0:2]))
#
# #compute Fourier transforms of x- and y- derivative kernels
# fx = np.asarray([0.5,0,-0.5]).reshape((1,3))
# fy = np.asarray([-0.5,0,0.5]).reshape((3,1))
# Fx = np.fft.fft2(kernpad(fx, nrm.shape[0:2]))
# Fy = np.fft.fft2(kernpad(fy, nrm.shape[0:2]))
# #compute conjugates
# Fx_conj = np.conj(Fx)
# Fy_conj = np.conj(Fy)
#
# #compute Fz. Remember to add a small number to the denominator.
# Fz = (Fx_conj*Gx + Fy_conj*Gy) / (np.abs(Fx)**2+np.abs(Fy)**2+lmda*np.abs(Fr)**2 + 10**(-12))
# Fz[0,0] = 0
#
# #invert transform to get the depth map Z
# Z = np.fft.ifft2(Fz)
# print(Z)
