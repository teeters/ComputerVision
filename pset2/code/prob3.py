### Default modules imported. Import more if you need to.
### DO NOT USE linalg.lstsq from numpy or scipy

import numpy as np
from skimage.io import imread, imsave

## Fill out these functions yourself


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns nrm:
#    nrm: HxWx3 Unit normal vector at each location.
#
# Be careful about division by zero at mask==0 for normalizing unit vectors.
def pstereo_n(imgs, L, mask):
    #convert images to grayscale
    grays = np.asarray([np.mean(img, 2) for img in imgs])
    #transpose so that each pixel location stores a list of its N values
    I = np.transpose(grays, (1,2,0))
    #compute two sides of equation
    #create image-sized matrix of copies of L so we can do operations at each pixel
    lt = np.empty(I.shape[0:2]+L.shape)
    lt[:,:] = L
    lti = lt*I[:,:,:,np.newaxis]
    lti = np.sum(lti, 2)
    ltl = np.empty(I.shape[0:2]+(3,3))
    ltl[:,:] = np.transpose(L).dot(L)
    n = np.linalg.solve(ltl,lti)
    nsums = np.sum(n**2,2)
    nsums = np.sqrt(nsums)
    n = n/nsums[:,:,np.newaxis]
    return n


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    nrm:  HxWx3 Unit normal vector at each location (from pstereo_n)
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns alb:
#    alb: HxWx3 RGB Color Albedo values
#
# Be careful about division by zero at mask==0.
def pstereo_alb(imgs, nrm, L, mask):
    #rho * sum_over_k (l_k^T n)^2 = sum_over_k l_k^Tn i_k
    #find l^Tn for every location in every image
    ln = nrm[np.newaxis, :,:,:] * L[:,np.newaxis, np.newaxis, :] #shape should be KxHxWx3
    ln = np.sum(ln,-1) #shape should now be KxHxW, with value l^tn at each location
    #square and sum over K
    ln_squared = np.sum(ln**2, 0)
    print(ln_squared.shape)

    #multiply ln by i for every color at every pixel in every image
    I = np.asarray(imgs)
    lni = I*ln[:,:,:,np.newaxis]
    #sum over K
    lni = np.sum(lni,0)
    print(lni.shape)
    rho = lni/ln_squared[:,:,np.newaxis]
    print(rho.shape)
    return rho

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

### Light directions matrix
L = np.float32( \
                [[  4.82962877e-01,   2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,   2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,   2.58819044e-01,   8.36516261e-01],
                 [ -5.00000060e-01,   0.00000000e+00,   8.66025388e-01],
                 [ -2.58819044e-01,   0.00000000e+00,   9.65925813e-01],
                 [ -4.37113883e-08,   0.00000000e+00,   1.00000000e+00],
                 [  2.58819073e-01,   0.00000000e+00,   9.65925813e-01],
                 [  4.99999970e-01,   0.00000000e+00,   8.66025448e-01],
                 [  4.82962877e-01,  -2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,  -2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,  -2.58819044e-01,   8.36516261e-01]])


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program


# Load image data
imgs = []
for i in range(L.shape[0]):
    imgs = imgs + [np.float32(imread(fn('inputs/phstereo/img%02d.png' % i)))/255.]

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = pstereo_n(imgs,L,mask)

nimg = nrm/2.0+0.5
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('outputs/prob3_nrm.png'),nimg)


alb = pstereo_alb(imgs,nrm,L,mask)

alb = alb / np.max(alb[:])
alb = clip(alb * mask[:,:,np.newaxis])

imsave(fn('outputs/prob3_alb.png'),alb)
