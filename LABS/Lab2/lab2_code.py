import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

img =  cv.imread("boat.png", cv.IMREAD_GRAYSCALE)
print(img.shape)
f = plt.figure(1)
ax =f.add_subplot(1,2,1)
ax.set_title('Original Image')
plt.imshow(img,cmap='gray')


# converting the type for the image and normalize by 255
img = img.astype(np.float32) / 255
ax = f.add_subplot(1,2,2)
ax.set_title('Normalized Image')
plt.imshow(img,cmap='gray')


plt.show()

# boat -2 printing pixel value
pixel = img[132,112]
print(f"Pixel value @ (132,112) {pixel}")

a ,b = 0,200
corner = img[a:b,a:b]
f = plt.figure(2)
plt.imshow(corner,cmap= 'gray')


plt.show()

#3 linear filtering
Ma = 1./9.*np.array([[1.,1.,1.,],[1.,1.,1.],[1.,1.,1.,]])
img_filtered = convolve2d(img,Ma,mode='same')
print("Filtered Image size/shape: ",img_filtered.shape)
plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')

plt.subplot(1,2,2)
plt.imshow(img_filtered,cmap='gray')


plt.show()


