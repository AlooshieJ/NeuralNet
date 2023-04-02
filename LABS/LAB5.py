import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train) , (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

#Q1) printing the shapes of CIFAR10 training and testing data
print("Train Data shapes: ")
print(x_train.shape, y_train.shape)
print("Test Data Shapes:")
print(x_test.shape,y_test.shape)


#Q2) 3x3 plot of first 9 images in training data
plt.figure(1)
plt.subplot(331)
plt.imshow(x_train[0])
plt.subplot(332)
plt.imshow(x_train[1])
plt.subplot(333)
plt.imshow(x_train[2])
plt.subplot(334)
plt.imshow(x_train[3])
plt.subplot(335)
plt.imshow(x_train[4])
plt.subplot(336)
plt.imshow(x_train[5])
plt.subplot(337)
plt.imshow(x_train[6])
plt.subplot(338)
plt.imshow(x_train[7])
plt.subplot(339)
plt.imshow(x_train[8])

# I might have this commented out to not show the grid of first
# images every time I run the code
#plt.show()

"""
The function takes two arguments since our data has two features: image and label. 
What it does is that it “standardizes” each image into having a zero mean, and a unity standard deviation.
"""
def preprocess_fn(image,label):
    image = tf.cast(image,dtype=tf.float32)
    image = tf.image.per_image_standardization(image)
    return (image,label)

def preprocess_new_fn(image,label):
    image = tf.cast(image,dtype=tf.float32)
    image = tf.image.per_image_standardization(image)
    #random horizontal flipping
    image = tf.image.random_flip_left_right(image)
    # random contrast Perturbation
    image = tf.image.random_contrast(image,0.2,0.5)
    return (image,label)

# Converting np arrays into dataset class from tf
x_train_ds = tf.data.Dataset.from_tensor_slices(x_train)
y_train_ds = tf.data.Dataset.from_tensor_slices(y_train)
train_ds = tf.data.Dataset.zip((x_train_ds,y_train_ds))


train_ds = train_ds.map(preprocess_fn)
# shuffle and batch the input data
BATCH_SIZE = 256
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).repeat()

# Q3)creating dataset for the test data.

x_test_ds = tf.data.Dataset.from_tensor_slices(x_test)
y_test_ds = tf.data.Dataset.from_tensor_slices(y_test)
test_ds = tf.data.Dataset.zip((x_test_ds,y_test_ds))
test_ds = test_ds.map(preprocess_fn)
test_ds = test_ds.batch(BATCH_SIZE)

#creating first 512 samples of test data
few_x_ds = tf.data.Dataset.from_tensor_slices(x_test[:512])
few_y_ds = tf.data.Dataset.from_tensor_slices(y_test[:512])
few_ds = tf.data.Dataset.zip((few_x_ds,few_y_ds))
few_ds = few_ds.map(preprocess_fn)
few_ds = few_ds.batch(BATCH_SIZE)


#Q4) implimenting the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(5,5),activation = 'relu',input_shape=(32,32,3)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32,(5,5),activation='relu',input_shape=(32,32,3)),
    tf.keras.layers.Conv2D(32,(5,5),activation='relu',input_shape=(32,32,3)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(10)
])

#Q5) the results will be obtained from model.fit() method
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss_fn,metrics=['sparse_categorical_accuracy'])
model.fit(train_ds,validation_data=test_ds,steps_per_epoch=50000//BATCH_SIZE,epochs=5,
          validation_steps=10000//BATCH_SIZE)

"""
# trying different data augmentation
new_train_ds = tf.data.Dataset.zip((x_train_ds,y_train_ds))
new_train_ds = new_train_ds.map(preprocess_new_fn)
new_train_ds = new_train_ds.shuffle(1000).batch(BATCH_SIZE).repeat()
print("training with augmented data ")
model.fit(new_train_ds,validation_data=test_ds,steps_per_epoch=50000//BATCH_SIZE,epochs =5,
          validation_steps =10000//BATCH_SIZE )

"""
# try:
#
#     model.save("Models/my_model")
# except:
#     print("COULD NOT SAVE")
#
# num_samples = 5000
# x_samples = x_test[:num_samples]
# y_samples = y_test[:num_samples]
#
#
# predictions = model.predict(x_samples)
#
# count = 0
# accuracy_count = 0
# for p in predictions:
#     x = np.argmax(p)
#     y = y_samples[count][0]
#     if x == y:
#         accuracy_count += 1
#     count += 1
# print(accuracy_count)
# print(accuracy_count / num_samples)
