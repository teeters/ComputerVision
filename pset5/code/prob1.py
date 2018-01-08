## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d as conv2

def get_cluster_centers(im,num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.
    cluster_centers = np.zeros((num_clusters,2),dtype='int')

    #calculate intervals for evenly spaced clusters
    H, W, C = im.shape
    dx = int(W/(np.sqrt(num_clusters)-1)) #luckily for me, k seems to always be a perfect square :)
    dy = int(H/(np.sqrt(num_clusters)-1))

    #get grid of evenly spaced coordinates
    # x_range = np.arange(0, W, d)
    # y_range = np.arange(0, H, d)
    # ycoords, xcoords = np.meshgrid(y_range, x_range, indexing='ij')
    # xcoords = xcoords.astype(int)
    # ycoords = ycoords.astype(int)
    yi, xi = np.indices((H,W))
    xcoords = xi[0:H:dy, 0:W:dx]
    ycoords = yi[0:H:dy, 0:W:dx]

    #get a 3x3 neighborhood around each coordinate
    n = np.asarray((-1,0,1))
    xn = np.repeat(xcoords[:,:,np.newaxis], 3,2)[:,:,:, np.newaxis] + n[np.newaxis,np.newaxis,np.newaxis,:]
    xn = xn.reshape(xcoords.shape+(9,))
    xn = np.maximum(0, np.minimum(xn, W-1))
    yn = np.repeat((ycoords[:,:,np.newaxis] + n[np.newaxis, np.newaxis, :])[:,:,:,np.newaxis], 3,3)
    yn = yn.reshape(ycoords.shape+(9,))
    yn = np.maximum(0, np.minimum(yn, H-1))

    #get point with minimum gradient within each neighborhood
    grads = get_gradients(im)
    nmin = np.argmin(grads[yn,xn], axis=2)
    yni, xni = np.indices(xcoords.shape)
    xmin = xn[yni, xni, nmin]
    ymin = yn[yni, xni, nmin]
    cluster_centers=np.concatenate((ymin.flatten()[:,np.newaxis], xmin.flatten()[:,np.newaxis]), 1)
    return cluster_centers

def slic(im,num_clusters,cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h,w,c = im.shape
    imy, imx = np.indices((h,w))
    sw = 5 #spatial weight
    iw = 1 #intensity weight
    s = int(np.sqrt(h*w/num_clusters)) #neighborhood to search around each centroid

    #initialize mindist to infinity for all pixels
    mindist = np.full((h,w), float('inf'))
    # keep track of assignment to each cluster
    cluster_ids = np.zeros((h,w))

    updating = True
    iters = 0
    while updating and iters<100:
        iters += 1
        #iterate over centers
        #for each center, calculate dist to pixels in 2sx2s window
        #print(cluster_centers)
        cluster_i = im[cluster_centers[:,0], cluster_centers[:,1]]
        for k in range(cluster_centers.shape[0]):
            cy, cx = cluster_centers[k]
            # get neighborhood indices
            y2 = min(h, cy+s); y1=max(0,cy-s)
            x2 = min(w, cx+s); x1=max(0,cx-s)
            yi, xi = np.indices((y2-y1,x2-x1))
            yi += y1; xi += x1

            # compute spatial and intensity distances
            sdists = (yi-cy)**2 + (xi-cx)**2
            idists = np.sum((im[yi,xi] - cluster_i[k])**2, axis=2)
            # update mindist using weights
            cdist = np.full(mindist.shape, float('inf'))
            cdist[y1:y2, x1:x2] = iw*idists+sw*sdists
            minidx = np.where(cdist<mindist)
            mindist[minidx] = cdist[minidx]
            cluster_ids[minidx] = k

        #now compute new centroids based on mean of points in clusters
        updating = False
        for k in range(cluster_centers.shape[0]):
            cy, cx = cluster_centers[k]
            #get indices for each point in cluster
            in_cluster = np.where(cluster_ids==k)
            numpoints = np.size(in_cluster)
            newx = int(np.sum(in_cluster[1])/numpoints)
            newy = int(np.sum(in_cluster[0])/numpoints)
            newi = (np.sum(im[in_cluster], axis=(0))/numpoints).astype(int)
            #continue iterating only if a centroid has moved:
            if np.sqrt((newx-cx)**2+(newy-cy)**2) > 1:
                updating = True

            cluster_centers[k,:] = (newy, newx)
            cluster_i[k] = newi

        iters += 1
    return cluster_ids

########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Use get_gradients (code from pset1) to get the gradient of your image when initializing your cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im,axis=2)
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    gx = conv2(im,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    gy = conv2(im,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    return np.sqrt(gx*gx+gy*gy)

# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im

# create an output image of our cluster centers
def create_centers_im(im,centers):
    for center in centers:
        im[center[0]-2:center[0]+2,center[1]-2:center[1]+2] = [255.,0.,255.]
    return im

im = np.float32(imread(fn('inputs/lion.jpg')))

num_clusters = [25,49,64,81,100]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im,num_clusters)
    imsave(fn('outputs/prob1a_' + str(num_clusters)+'_centers.jpg'),normalize_im(create_centers_im(im.copy(),cluster_centers)))
    out_im = slic(im,num_clusters,cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1,np.float32(out_im.flatten())/float(num_clusters)))[:,0:3]
    dimg = dimg.reshape([out_im.shape[0],out_im.shape[1],3])
    imsave(fn('outputs/prob1b_'+str(num_clusters)+'.jpg'),normalize_im(dimg))
