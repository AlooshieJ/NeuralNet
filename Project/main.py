import pandas as pd
import os
import matplotlib.pyplot as plt

fire_data_path = 'Fire_data/fire/'
nofire_data_path = 'Fire_data/nofire/'
fire = []
nofire=[]

#loading in fire data
for im in os.listdir('Fire_data/fire'):
    im = plt.imread(fire_data_path+im)
    fire.append(im)
for im in os.listdir('Fire_data/nofire'):
    im = plt.imread(nofire_data_path+im)
    nofire.append(im)

fire_len = len(fire)
nofire_len = len(nofire)
print(f"fire elements:{fire_len}")
print(f"no fire elements:{nofire_len}")
im = fire[300]

print(im.shape)

plt.imshow(im)

plt.show()

