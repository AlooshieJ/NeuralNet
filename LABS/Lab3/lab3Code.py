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