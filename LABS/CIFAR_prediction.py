import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
label_names =['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
(x_train,y_train) , (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

num_samples = 512
x_samples = x_test[:num_samples]
y_samples = y_test[:num_samples]
model = tf.keras.models.load_model("Models/my_model")
print(model.history)

predictions = model.predict(x_samples)
# dictonary for labels, increment count based on prediciton
labels= {
'airplane':0,
    'automobile': 0,
    'bird': 0,
    'cat': 0,
    'deer': 0,
    'dog': 0,
    'frog':0,
    'horse':0,
    'ship':0,
    'truck':0
}
count = 0
accuracy_count = 0
for p in predictions:
    x = np.argmax(p)
    y = y_samples[count][0]
    # print(x,y)
    if x == y:
        labels[label_names[y]] += 1
        accuracy_count+= 1
    count += 1
print(accuracy_count)
print(accuracy_count / num_samples)

