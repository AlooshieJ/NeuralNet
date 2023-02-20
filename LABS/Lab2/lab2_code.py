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



# boat -2 printing pixel value
pixel = img[132,112]
print(f"Pixel value @ (132,112) {pixel}")
#3 corner of image
a ,b = 0,200
corner = img[a:b,a:b]
f = plt.figure(2)
plt.imshow(corner,cmap= 'gray')


#3 linear filtering
Ma = 1./9.*np.array([[1.,1.,1.,],[1.,1.,1.],[1.,1.,1.,]])
img_filtered = convolve2d(img,Ma,mode='same')
print("Filtered Image size/shape: ",img_filtered.shape)
f3 = plt.figure(3)
plt.subplot(3,2,1)
plt.imshow(img,cmap='gray')
plt.title('Original')

plt.subplot(3,2,2)
plt.imshow(img_filtered,cmap='gray')
plt.title('avg. filter')

# 4)
cv.imwrite("Filtered_boat.png",img_filtered)

#5) 3x3 difference filter
mb = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
print(mb)
img_diff = convolve2d(img,mb,mode="same")
print(img_diff.shape)

img_diff += .5
plt.subplot(3,2,3)
plt.imshow(img_diff,cmap='gray')
plt.title('diff. filter')
#6) horizontal edges
horizontal_filter = np.array([[-1,2,-1]])
horiz = convolve2d(img,horizontal_filter,mode='same')
horiz += 0.5
plt.subplot(3,2,4)
plt.imshow(horiz,cmap='gray')
plt.title('Horizontal Filter')
#7) vertical Edges
vertical_filter = horizontal_filter.transpose()
verts = convolve2d(img,vertical_filter,mode='same')
verts += 0.5
plt.subplot(3,2,5)
plt.imshow(verts,cmap='gray')
plt.title('Vertical filter')
plt.show()

