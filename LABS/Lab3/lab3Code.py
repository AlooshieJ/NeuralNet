import gzip
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"
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




train_ims, train_lbs, test_ims, test_lbs = load_data()

print("Train ims shape:",train_ims.shape)
print("Train lbs shape:",train_lbs.shape)
print("Test ims shape:",test_ims.shape)
print("Test lbs shape:",test_lbs.shape)

#5th sample from the data set from train_ims & train_lbs

sample = train_ims[4]
l_sample = train_lbs[4]
t_img = np.reshape(sample, (NROWS, NCOLS))
plt.figure(1)
plt.imshow(t_img)
print("sample label:",l_sample)

# 5th sample from data set of test ims & test lbs
sample = test_ims[4]
l_sample = test_lbs[4]
t_img = np.reshape(sample, (NROWS, NCOLS))
plt.figure(2)
plt.imshow(t_img)
print("sample label:",l_sample)


# Q4) only using samples of class 0 & 1

mask = np.logical_or(train_lbs==0, train_lbs==1)
train_ims = train_ims[mask,:]
train_lbs = train_lbs[mask]

print("filtering out all nums except 0&1 from train data:")
print(train_ims.shape,train_lbs.shape)

#Q5) doing the same filtering on the test data
mask = np.logical_or(test_lbs==0, test_lbs==1)
test_ims = test_ims[mask,:]
test_lbs = test_lbs[mask]

print("filtering out all nums except 0&1 from test data:")
print(test_ims.shape,test_lbs.shape)


#Q6) creating a validation data set

val_ims = train_ims[:int(0.8*train_ims.shape[0]),:]
train_ims = train_ims[int(0.8*train_ims.shape[0]):,:]

val_lbs = train_lbs[:int(0.8*train_lbs.shape[0])]
train_lbs = train_lbs[int(0.8*train_lbs.shape[0]):]

print("training samples:")
print(train_ims.shape,train_lbs.shape)

print("validation samples:")
print(val_ims.shape,val_lbs.shape)

#Q7 type casting the datasets to float 32
val_ims.astype(np.float32)
val_lbs.astype(np.float32)

train_ims.astype(np.float32)
train_lbs.astype(np.float32)

test_ims.astype(np.float32)
test_lbs.astype(np.float32)

#Q8) converting labels values from {0,1} to {-1,1} no loop
#
# val_lbs
# train_lbs
# test_lbs

plt.show()