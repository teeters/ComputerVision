## Default modules imported. Import more if you need to.

import numpy as np


## Fill out these functions yourself

# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')
#
# Can use np.linalg.svd
def getH(pts):
    #convert 2d points to homogeneous coordinates by adding a column of 1s
    pts1 = np.concatenate((pts[:, 0:2], np.ones((pts.shape[0], 1))), axis=1)
    pts2 = np.concatenate((pts[:,2:4], np.ones((pts.shape[0], 1))), axis=1)

    #iterate over points (assuming small number),
    #calculate the Ai matrix for each and add it to the list
    #assuming p is on the left and p' is on the right
    A_components = []
    for p, p_prime in zip(pts1, pts2):
        p1, p2, p3 = p_prime
        zeros = np.zeros(3)
        Ai = np.asarray([
            np.concatenate((zeros, -p3*p, p2*p)),
            np.concatenate((p3*p, zeros, -p1*p)),
            np.concatenate((-p2*p, p1*p, zeros))
        ])
        A_components.append(Ai)
    A = np.vstack(A_components)

    #solve Ah=0 using SVD
    u, s, v = np.linalg.svd(A)
    #get row of v corresponding to smallest singular value
    i = np.argmin(np.abs(s))
    h = v[i,:]
    return h.reshape(3,3)


# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.
def splice(src,dest,dpts):
    #get homography mapping from dest points to source points
    src_H, src_W, _ = src.shape
    src_pts = np.float32([[0,0], [src_W-1,0], [0,src_H-1], [src_W-1,src_H-1]])
    pts = np.concatenate((dpts, src_pts), axis=1)
    H = getH(pts)

    #create matrices of coordinates in destination quadrilateral
    #note that the corresponding source points for some will lie outside the source image
    min_x = np.min(dpts[:,0])
    min_y = np.min(dpts[:,1])
    max_x = np.max(dpts[:,0])
    max_y = np.max(dpts[:,1])
    dest_xrange = np.arange(min_x, max_x+1)
    dest_yrange = np.arange(min_y, max_y+1)
    xx = np.repeat(dest_xrange, np.size(dest_yrange))
    yy = np.tile(dest_yrange, np.size(dest_xrange))
    dest_coords = np.concatenate((xx[:,np.newaxis], yy[:,np.newaxis]), axis=1)

    #apply homography to get corresponding source coordinates
    src_coords = apply_homography(dest_coords, H).astype(int)
    dest_coords = dest_coords.astype(int)
    #eliminate locations where coords are out of range
    lower = np.asarray([0,0])
    upper = np.asarray([src_W, src_H])
    filter = np.logical_and(src_coords>=lower, src_coords<upper)
    filter = np.all(filter,axis=1)
    src_coords = src_coords[filter]
    dest_coords = dest_coords[filter]

    #assign pixels at src_coords to locations at dest_coords
    modified = dest
    modified[dest_coords[:,1], dest_coords[:,0]] = src[src_coords[:,1], src_coords[:,0]]
    return modified

def apply_homography(pts, H):
    '''Returns the points transformed by H, still in (x,y) format'''
    pts_aug = np.concatenate((pts, np.ones((pts.shape[0], 1))), 1)
    #for each row in pts, multiply that row pointwise with each row in h
    multiplied = H[np.newaxis, :,:] * pts_aug[:,np.newaxis,:]
    summed = np.sum(multiplied, axis=2)
    result = (summed/summed[:,2][:,np.newaxis])[:,0:2]
    return result


########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

#test homography
# test_src = np.asarray([[3,3], [7,3], [3,7], [7,7]])
# test_dest = np.asarray([[0,0], [2,0], [0,2], [2,2]])
# test_h = getH(np.concatenate((test_dest, test_src), 1))
# print("Test homography projection")
# correct_h = np.asarray([[2,0,3], [0,2,3], [0,0,1]])
# print(apply_homography(test_dest, correct_h))

simg = np.float32(imread(fn('inputs/p4src.png')))/255.
dimg = np.float32(imread(fn('inputs/p4dest.png')))/255.
dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded

comb = splice(simg,dimg,dpts)

imsave(fn('outputs/prob4.png'),comb)
