"""
491 Homework 5: 3 - layer connected Network using the pytorch framework,
classifying mnist data
"""
import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


#------------------------ loading mnist data in (from lab 3) ------------------------ #
# DATASET_DIR = "dataset"
DATASET_DIR = "/Users/alijafar/Desktop/PycharmCode/NeuralNet/LABS/Lab3/dataset"
MNIST_TRAIN_IMS_GZ = os.path.join(DATASET_DIR, "train-images-idx3-ubyte.gz")
MNIST_TRAIN_LBS_GZ = os.path.join(DATASET_DIR, "train-labels-idx1-ubyte.gz")
MNIST_TEST_IMS_GZ = os.path.join(DATASET_DIR, "t10k-images-idx3-ubyte.gz")
MNIST_TEST_LBS_GZ = os.path.join(DATASET_DIR, "t10k-labels-idx1-ubyte.gz")

NROWS = 28
NCOLS = 28
def load_data():
    print("Unpacking training images ...")
    with gzip.open(MNIST_TRAIN_IMS_GZ, mode='rb') as f:
        magic_num, train_sz, nrows, ncols = struct.unpack('>llll', f.read(16))
        print("magic number: %d, num of examples: %d, rows: %d, columns: %d" % (magic_num, train_sz, nrows, ncols))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * train_sz * nrows * ncols, data_bn)
        train_ims = np.asarray(data)
        train_ims = train_ims.reshape(train_sz, nrows * ncols)
    print("~" * 5)

    print("Unpacking training labels ...")
    with gzip.open(MNIST_TRAIN_LBS_GZ, mode='rb') as f:
        magic_num, train_sz = struct.unpack('>ll', f.read(8))
        print("magic number: %d, num of examples: %d" % (magic_num, train_sz))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * train_sz, data_bn)
        train_lbs = np.asarray(data)
    print("~" * 5)

    print("Unpacking test images ...")
    with gzip.open(MNIST_TEST_IMS_GZ, mode='rb') as f:
        magic_num, test_sz, nrows, ncols = struct.unpack('>llll', f.read(16))
        print("magic number: %d, num of examples: %d, rows: %d, columns: %d" % (magic_num, train_sz, nrows, ncols))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * test_sz * nrows * ncols, data_bn)
        test_ims = np.asarray(data)
        test_ims = test_ims.reshape(test_sz, nrows * ncols)
    print("~" * 5)

    print("Unpacking test labels ...")
    with gzip.open(MNIST_TEST_LBS_GZ, mode='rb') as f:
        magic_num, test_sz = struct.unpack('>ll', f.read(8))
        print("magic number: %d, num of examples: %d" % (magic_num, train_sz))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * test_sz, data_bn)
        test_lbs = np.asarray(data)
    print("~" * 5)
    return train_ims, train_lbs, test_ims, test_lbs

#------------------------ loading data in ------------------------ #


train_ims, train_lbs, test_ims, test_lbs = load_data()

train_ims,train_lbs,test_ims,test_lbs =\
torch.tensor(train_ims,dtype=torch.float32),\
torch.tensor(train_lbs),\
torch.tensor(test_ims,dtype=torch.float32),\
torch.tensor(test_lbs)
print("Train ims shape:",train_ims.shape,type(train_ims),train_ims.dtype)
print("Train lbs shape:",train_lbs.shape,type(train_lbs))
print("Test ims shape:",test_ims.shape,type(test_ims))
print("Test lbs shape:",test_lbs.shape,type(test_lbs))

#Defining device:
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# class for the liner network model
class linear3layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear_stack = nn.Sequential(
            nn.Linear(NROWS*NCOLS,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        return self.Linear_stack(x)
        pass

lr = 1e-3
batch_size = 64

# invocing network class
model = linear3layer().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr)

# length of image data set
num_train_ims = len(train_ims)
num_test_ims = len(test_ims)
def train_loop(data,data_lbs,model,loss_fn,optimizer):
    size = len(data)
    model.train()
    for idx in range(len(data)):
        X = data[idx,:].to(device)
        y= data_lbs[idx].to(device)
        #calculating prediction
        pred = model(X)
        loss=  loss_fn(input=pred,target=y)
        #backprop part
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10000 == 0:
            loss,current = loss.item(),(idx)#*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(data,data_lbs,model,loss_fn):
    size = len(data)
    num_batches = len(data)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for idx in range(len(data)):
            X = data[idx, :].to(device)
            y = data_lbs[idx].to(device)
            # calculating prediction
            pred = model(X)
            test_loss += loss_fn(input=pred, target=y).item()

            #print(torch.argmax(pred),y)
            if torch.argmax(pred) == y:
                correct+=1
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
# running the training and testing on the network model
epochs = 1
for i in range(epochs):
    print(f"Epoch {i+1}\n--------------")
    train_loop(train_ims,train_lbs,model,loss_fn,optimizer)
    test_loop(test_ims,test_lbs,model,loss_fn)
    print("DONE!")



#torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

weight_img = model.state_dict().keys()['weight'].reshape((NROWS,NCOLS))
plt.figure(1)
plt.imshow(weight_img)
plt.show()