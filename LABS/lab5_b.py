import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Loading CIFAR-10 DATASET")
(x_train,y_train) , (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

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

BATCH_SIZE = 256
print("creating train dataset")
# Converting np arrays into dataset class from tf
x_train_ds = tf.data.Dataset.from_tensor_slices(x_train)
y_train_ds = tf.data.Dataset.from_tensor_slices(y_train)
train_ds = tf.data.Dataset.zip((x_train_ds,y_train_ds))

train_ds = train_ds.map(preprocess_new_fn)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).repeat()

print("creating test dataset")
#creating the test dataset
x_test_ds = tf.data.Dataset.from_tensor_slices(x_test)
y_test_ds = tf.data.Dataset.from_tensor_slices(y_test)
test_ds = tf.data.Dataset.zip((x_test_ds,y_test_ds))
test_ds = test_ds.map(preprocess_fn)
test_ds = test_ds.batch(BATCH_SIZE)

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

print("training the model")
#training the model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss_fn,metrics=['sparse_categorical_accuracy'])
model.fit(train_ds,validation_data=test_ds,steps_per_epoch=50000//BATCH_SIZE,epochs =5,
          validation_steps =10000//BATCH_SIZE )