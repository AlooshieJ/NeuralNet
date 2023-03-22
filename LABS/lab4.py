"""
Implementing a Shallow NN using tensorflow and Keras framework
Using the MNIST data set for characterizing hand written digits

By: Ali Jafar
UIN: 669430206

"""

from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train , x_test  = x_train/255.0 , x_test/255.0

#Q1) printing the shapes of the variables
print("Mnist Train Data Shape:")
print(f"Train Data:{x_train.shape} Train Labels: {y_train.shape}")

print("MNIST Test Data Shape:")
print(f"Test Data:{x_test.shape} Test Labels: {y_test.shape}")
#Q2) split data into 80/20 training /validation
print("splitting Train data into 80%/20% Training/Validation")
ratio_len = .2 * len(x_train)
x_val = x_train[:int(ratio_len),:]
y_val = y_train[:int(ratio_len)]

x_train = x_train[int(ratio_len):,:]
y_train = y_train[int(ratio_len):]

print("New Shapes:")
print(f"Train Data:{x_train.shape} Train Labels: {y_train.shape}")
print(f"Validation Data:{x_val.shape} Validation Labels: {y_val.shape}")

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

#Q4) output shape and entire model shape
print(model.output.get_shape().as_list())
model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
#Q5) training the model
print("Training Model")
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=5)
#Q6) model over testing data
print("Evaluation over Test data")
model.evaluate(x_test,y_test,verbose=2)

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(200,activation='relu'),
    tf.keras.layers.Dense(200,activation='relu'),
    tf.keras.layers.Dense(10)
])

print("---- Modle #2 ----")
print("Model Summary:")
model2.summary()
model2.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
model2.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=5)
model2.evaluate(x_test,y_test,verbose=2)