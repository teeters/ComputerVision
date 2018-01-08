import numpy as np
np.random.seed(0)

from os.path import normpath as fn
from time import time

import edf ## This will be your code

# Load data
data = np.load(fn('inputs/mnist_26k.npz'))

train_im = np.float32(data['im_train'])/255.-0.5
train_im = np.reshape(train_im,[-1,28,28,1])
train_lb = data['lbl_train']

val_im = np.float32(data['im_val'])/255.-0.5
val_im = np.reshape(val_im,[-1,28,28,1])
val_lb = data['lbl_val']


#######################################

# Inputs and parameters
inp = edf.Value()
lab = edf.Value()

K1 = edf.Param()
B1 = edf.Param()
W2 = edf.Param()
B2 = edf.Param()


# Model
y = edf.conv2(inp,K1)
y = edf.down2(y); y = edf.down2(y);
y = edf.add(y,B1)
y = edf.RELU(y)

y = edf.flatten(y)

y = edf.matmul(y,W2)
y = edf.add(y,B2) # This is our final prediction


# Cross Entropy of Soft-max
loss = edf.smaxloss(y,lab)
loss = edf.mean(loss)

# Accuracy
acc = edf.accuracy(y,lab)
acc = edf.mean(acc)

###################################

# Init Weights
def xavier(shape):
    sq = np.sqrt(3.0/np.prod(shape[:-1]))
    return np.random.uniform(-sq,sq,shape)

C1 = 64
K1.set(xavier((8,8,1,C1)))
B1.set(np.zeros((C1)))

W2.set(xavier((C1*36,10)))
B2.set(np.zeros((10)))


# Training loop

BSZ=50
lr=0.0001

NUM_EPOCH=4 ## Will be slow
DISPITER=50
batches = range(0,len(train_lb)-BSZ+1,BSZ)

## Implement Momentum and uncomment following line
edf.init_momentum()


niter=0; avg_loss = 0.; avg_acc = 0.
for ep in range(NUM_EPOCH+1):

    # As we train, let's keep track of val accuracy
    vacc = 0.; vloss = 0.; viter = 0
    for b in range(0,len(val_lb)-BSZ+1,BSZ):
        inp.set(val_im[b:b+BSZ,...]); lab.set(val_lb[b:b+BSZ])
        edf.Forward()
        viter = viter + 1;vacc = vacc + acc.top;vloss = vloss + loss.top
    vloss = vloss / viter; vacc = vacc / viter * 100
    print("%09d: #### %d Epochs: Val Loss = %.3e, Accuracy = %.2f%%" % (niter,ep,vloss,vacc))
    if ep == NUM_EPOCH:
        break

    # Shuffle Training Set
    idx = np.random.permutation(len(train_lb))

    # Train one epoch
    for b in batches:
        # Load a batch
        inp.set(train_im[idx[b:b+BSZ],...])
        lab.set(train_lb[idx[b:b+BSZ]])

        edf.Forward()
        avg_loss = avg_loss + loss.top; avg_acc = avg_acc + acc.top;
        niter = niter + 1
        if niter % DISPITER == 0:
            avg_loss = avg_loss / DISPITER; avg_acc = avg_acc / DISPITER * 100
            print("%09d: Training Loss = %.3e, Accuracy = %.2f%%" % (niter,avg_loss,avg_acc))
            avg_loss = 0.; avg_acc = 0.;

        edf.Backward(loss)
        #edf.SGD(lr)
        # Replace previous line with following
        edf.momentum(lr,0.9)

