import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img =  cv.imread("boat.png", cv.IMREAD_GRAYSCALE)
print(img.shape)
f = plt.figure(1)
ax =f.add_subplot(1,2,1)
ax.set_title('Original Image')
plt.imshow(img, cmap='gray')


# converting the type for the image and normalize by 255
img = img.astype(np.float32) / 255
ax = f.add_subplot(1,2,2)
ax.set_title('Normalized Image')
plt.imshow(img,cmap='gray')

# boat -2 printing pixel value
pixel = img[132,112]
print(f"Pixel value @ (132,112) {pixel}")

a ,b = 100,75
corner = img[a:b,a:b]
ax = f.add_subplot(2,1,1)
ax.set_title('top-left corner')
plt.imshow(corner,cmap="gray")
plt.show()


