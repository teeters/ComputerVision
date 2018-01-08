### EDF --- An Autograd Engine for instruction
## (based on joint discussions with David McAllester)

import numpy as np
from scipy.signal import convolve2d as conv2
import math

# Global list of different kinds of components
ops = []
params = []
values = []


# Global forward
def Forward():
    for c in ops: c.forward()

# Global backward
def Backward(loss):
    for c in ops:
        c.grad = np.zeros_like(c.top)
    for c in params:
        c.grad = np.zeros_like(c.top)

    loss.grad = np.ones_like(loss.top)
    for c in ops[::-1]: c.backward()

# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr*p.grad


## Fill this out
def init_momentum():
    for p in params:
        p.momentum = 0


## Fill this out
def momentum(lr,mom=0.9):
    for p in params:
        p.top = p.top - lr*(p.grad+mom*p.momentum)
        p.momentum = p.grad + mom*p.momentum


###################### Different kinds of nodes

# Values (Inputs)
class Value:
    def __init__(self):
        values.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

# Parameters (Weights we want to learn)
class Param:
    def __init__(self):
        params.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()


### Operations

# Add layer (x + y) where y is same shape as x or is 1-D
class add:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad

        if self.y in ops or self.y in params:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
                self.y.grad = self.y.grad + ygrad

# Matrix multiply (fully-connected layer)
class matmul:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top,self.y.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in ops or self.y in params:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)


# Rectified Linear Unit Activation
class RELU:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)


# Reduce to mean
class mean:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))



# Soft-max + Loss (per-row / training example)
class smaxloss:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        y = self.x.top
        y = y - np.amax(y,axis=1,keepdims=True)
        yE = np.exp(y)
        yS = np.sum(yE,axis=1,keepdims=True)
        y = y - np.log(yS); yE = yE / yS

        truey = np.int64(self.y.top)
        self.top = -y[range(len(truey)),truey]
        self.save = yE

    def backward(self):
        if self.x in ops or self.x in params:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save
        # No backprop to labels!

# Compute accuracy (for display, not differentiable)
class accuracy:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        pass


# Downsample by 2
class down2:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = self.x.top[:,::2,::2,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::2,::2,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class flatten:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)

# Convolution Layer
## Fill this out
class conv2:

    def __init__(self,x,k):
        ops.append(self)
        self.x = x
        self.k = k

    def naive_forward(self):
        '''Slow but safe convolution to check values'''
        B, H, W, C = self.x.top.shape
        kH, kW, Cin, Cout = self.k.top.shape
        conv = np.zeros((B, H-kH+1, W-kW+1, Cout))
        for b in range(B):
            for y in range(H-kH+1):
                for x in range(W-kW+1):
                    ix1 = x; ix2 = x+kW
                    iy1 = y; iy2 = y+kH
                    val = np.sum( self.x.top[b,iy1:iy2, ix1:ix2, :, np.newaxis] * self.k.top, axis=(-2, 0,1))
                    conv[b,y,x] = val
        return conv



    def forward(self):
        #can't quite use built-in conv functions due to batched images
        #so we'll iterate over the kernel and add up subarrays
        #remember to sum input channels into output channels
        B, H, W, C = self.x.top.shape
        kH, kW, Cin, Cout = self.k.top.shape
        self.top = np.zeros((B, H-kH+1, W-kW+1, Cout))
        for ky, kx in np.ndindex((kH, kW)):
            ix1 = kx; ix2 = W-kW+kx+1;
            iy1 = ky; iy2 = H-kH+ky+1;
            self.top += np.sum(self.k.top[ky,kx]*self.x.top[:,iy1:iy2,ix1:ix2:,:,np.newaxis], axis=3)

    def backward(self):
        B, H, W, C = self.x.top.shape
        kH, kW, Cin, Cout = self.k.top.shape
        if self.x in ops or self.x in params:
            for ky, kx in np.ndindex(self.k.top.shape[0:2]):
                y1=ky; y2=H-kH+ky+1
                x1=kx; x2=W-kW+kx+1
                delta = np.sum( self.k.top[ky,kx]*self.grad[:, :, :, np.newaxis, :], axis=-1)
                self.x.grad[:, y1:y2, x1:x2, :] += delta

        if self.k in ops or self.k in params:
            for ky, kx in np.ndindex(self.k.top.shape[0:2]):
                y1=ky; y2=H-kH+ky+1
                x1=kx; x2=W-kW+kx+1
                #for each k[ky, kx, cin, cout], sum over b, y, and x in the input
                delta = np.sum(self.grad[ :, :, :, np.newaxis, :] * self.x.top[:,y1:y2, x1:x2, :, np.newaxis], axis=(0,1,2))
                self.k.grad[ky,kx] += delta

class conv2down:

    def __init__(self,x,k,s):
        #s is for stride
        ops.append(self)
        self.x = x
        self.k = k
        self.s = s

    def forward(self):
        B, H, W, C = self.x.top.shape
        kH, kW, Cin, Cout = self.k.top.shape
        top_shape = self.x.top[:,0:H-kH+1:self.s, 0:W-kW+1:self.s].shape[0:3] + (Cout,)
        self.top = np.zeros(top_shape)
        for ky, kx in np.ndindex((kH, kW)):
            ix1 = kx; ix2 = W-kW+kx+1;
            iy1 = ky; iy2 = H-kH+ky+1;
            self.top += np.sum(self.k.top[ky,kx]*self.x.top[:, iy1:iy2:self.s, ix1:ix2:self.s, :,np.newaxis], axis=3)

    def backward(self):
        B, H, W, C = self.x.top.shape
        kH, kW, Cin, Cout = self.k.top.shape
        if self.x in ops or self.x in params:
            grad_trans = np.repeat(np.repeat(self.grad, self.s,1), self.s,2)
            for ky, kx in np.ndindex(self.k.top.shape[0:2]):
                y1=ky; y2=H-kH+ky+1
                x1=kx; x2=W-kW+kx+1
                delta = np.sum( self.k.top[ky,kx,:,:]*self.grad[:, :, :, np.newaxis, :], axis=-1)
                self.x.grad[:,y1:y2:self.s, x1:x2:self.s, :] += delta

        if self.k in ops or self.k in params:
            grad_trans = np.repeat(np.repeat(self.grad, self.s, 1), self.s,2)
            for ky, kx in np.ndindex(self.k.top.shape[0:2]):
                y1=ky; y2=H-kH+ky+1
                x1=kx; x2=W-kW+kx+1
                delta = np.sum(self.grad[ :, :, :, np.newaxis, :] * self.x.top[:,y1:y2:self.s, x1:x2:self.s, :, np.newaxis], axis=(0,1,2))
                self.k.grad[ky,kx] += delta
