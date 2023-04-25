"""
designing LeNet-5 using Tensorflow api
to classify MNIST dataset
Now using dilated convolution instead of average pooling
ECE 491 : intro to NN
Ali Jafar
uin:669430206
"""
import tensorflow as tf
import numpy as np

print("Loading MNIST data")
(x_train,y_train),(x_test,y_test) =tf.keras.datasets.mnist.load_data()

print("train Data:")
print(x_train.shape,y_train.shape)
print("test Data:")
print(x_test.shape,y_test.shape)

x_train = x_train[:,:,:,np.newaxis]
x_test  = x_test[:,:,:,np.newaxis]

# defining the model
model = tf.keras.models.Sequential([
    #tf.keras.layers.Conv2D(6,(5,5),activation='tanh',input_shape=x_train.shape[1:],padding='same'),
    #tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
    #tf.keras.layers.Conv2D(16,(10,10),activation='relu',padding='valid'),
    #tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(6,(5,5),dilation_rate=(2,2),activation='tanh',input_shape=x_train.shape[1:],padding='same'),
    tf.keras.layers.Conv2D(16,(10,10),dilation_rate=(2,2),activation='relu',padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120,activation='tanh'),
    tf.keras.layers.Dense(84,activation='tanh'),
    tf.keras.layers.Dense(10,activation='softmax')

])


loss_fn =  tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer='adam',loss = loss_fn,metrics = ['sparse_categorical_accuracy'])
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5)

print(history.history)
