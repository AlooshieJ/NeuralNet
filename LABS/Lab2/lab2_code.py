import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img =  cv.imread("boat.png", cv.IMREAD_GRAYSCALE)
print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()
