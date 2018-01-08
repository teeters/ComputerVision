## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input 8-bit grayscale image
# Return equalized image with intensities from 0-255
def histeq(X):
    #Compute cdf of pixel frequencies.
    #Iterate through the pixels in sorted order, adding the frequency
    #of the previous intensity to the current one. Should be faster
    #than making a second pass.

    h, w = img.shape
    values, counts = np.unique(img, return_counts=True)
    p = np.zeros(256)
    for i, v in np.ndenumerate(values):
        p[v] = counts[i]
        if v > 0:
            p[v] += p[v-1]
    p = p / img.size
    #p[i] should now = frequency of pixels with intensity i or less

    #use frequencies to transform image
    Y = np.empty((h, w), dtype=int);
    for i in range(h):
        for j in range(w):
            Y[i][j] = int(p[ X[i][j] ] * 255)
    return Y


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = imread(fn('inputs/p2_inp.png'))

out = histeq(img)

out = np.maximum(0,np.minimum(255,out))
out = np.uint8(out)
imsave(fn('outputs/prob2.png'),out)
