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

def sign(n):
    if n>=0:
        return 1
    else:
        return -1
def RELU(n):
    if n <=0:
        return -1
    return n
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

test_lbs= np.where(test_lbs == 0 , -1,test_lbs)
train_lbs = np.where(train_lbs == 0 , -1,train_lbs)
val_lbs = np.where(val_lbs == 0 , -1,val_lbs)

# #printing to test if they changed
# for l in test_lbs : print(l,end=" ")
# print()
# for l in train_lbs : print(l,end=" ")
# print()
# for l in val_lbs : print(l,end=" ")


#Q9)
#weights = np.random.normal(0.0, 1.0, size=(NROWS*NCOLS))
weights = np.ones(NROWS*NCOLS)
#weights = np.zeros(NROWS*NCOLS)
num_training_samples = len(train_ims)
num_val_samples = len(val_ims)
print(num_training_samples)
eta = 1#this is a scalar variable for the learning rate, choose a suitable value

# the training function will return the weight as a numpy array
def train_loop(train_data,train_label,val_data,val_label,eta,weight):
    pass


for idx in range(num_training_samples):
    #read the i-th image
    x  = train_ims[idx,:]
    #read the i-th label
    y_true = train_lbs[idx]
    y_pred = sign(np.dot(weights.T,x))
    #error = 1/2*((y_true-y_pred)**2)
    error = .5*(y_true-y_pred)
    update = eta*error*x
    weights +=update

  #every 100 step we want to check the accuracy over the validation data
    acc_count = 0 #we will store the number of correct predictions
    if idx%100==0:
        for val_idx in range(num_val_samples):
            x = val_ims[val_idx,:]
            y_true = val_lbs[val_idx]
            #predict the label of the sample
            val_pred = sign(np.dot(weights.T,x))
            #val_pred = RELU(np.dot(weights.T,x))

            #if prediction is correct, increase the counter
            if val_pred == y_true:
                acc_count+=1

    accuracy = (acc_count*100.)/num_val_samples
    print("step:%d, acc:%.2f"%(idx, accuracy))
    #if accuracy is above 0.90, terminate by using “break”
    if accuracy > 90 :
        break

# Q12 accuracy with test data
print("-----NOW RUNNING ON TEST DATA-----")
acc_count = 0
num_test_samples = len(test_ims)
for test_idx in range(num_test_samples):
    x = test_ims[test_idx,:]
    y_true = test_lbs[test_idx]
    val_pred = sign(np.dot(weights.T,x))

    if val_pred == y_true:
        acc_count+=1
accuracy = (acc_count * 100.)/num_test_samples
    # print("Step: %d, acc:%.2f"%(test_idx,accuracy))
print("After testing with {} images, accuracy:{:.2f}%".format(num_test_samples,accuracy))

# Q13 displaying weights vector
weights_img = weights.reshape((NROWS,NCOLS))
#weights_img *=2
plt.figure(3)
plt.imshow(weights_img)


plt.show()
