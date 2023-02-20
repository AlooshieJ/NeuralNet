import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


img = cv.imread("Barbara.jpg")
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
print(img.shape)

f = plt.figure(1)
plt.imshow(img)

#1 - printing pixel values at (132,112)
pixel = img[132,112]
print(pixel)

#2 - corner of image , where img is 512x512
a,b = 0,128
corner = img[a:b,a:b]
f2 = plt.figure(2)
plt.imshow(corner)
print("top-left barb shape/size:",corner.shape)
corner = cv.cvtColor(corner,cv.COLOR_RGB2BGR)
cv.imwrite("topleft_barb.jpeg",corner)

#image now in RGB format
imgR = img[:,:,0]
imgG = img[:,:,1]
imgB = img[:,:,2]
print("shapes R,g,b",imgR.shape,imgG.shape,imgB.shape)

#3 - applying linear filtering on barbara
ma = 1./9.*np.array([[1.,1.,1.,],[1.,1.,1.],[1.,1.,1.,]])

img_filteredR = convolve2d(imgR,ma,mode="same")
img_filteredG = convolve2d(imgG,ma,mode="same")
img_filteredB = convolve2d(imgB,ma,mode="same")
img_linear = np.dstack( (img_filteredR,img_filteredG,img_filteredB) ).astype(np.uint8)

print("linear filtered image shape:",img_linear.shape)
plt.figure(3)
plt.imshow(img_linear)
# 4 writing the image
img_linear = cv.cvtColor(img_linear,cv.COLOR_RGB2BGR)
cv.imwrite("Linear_Barb.jpeg",img_linear)


#5 applying difference filter
mb = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

diff_filterR= convolve2d(imgR,mb,mode="same")
diff_filterG= convolve2d(imgG,mb,mode="same")
diff_filterB= convolve2d(imgB,mb,mode="same")
img_diff = np.dstack( (diff_filterR,diff_filterG,diff_filterB) ).astype(np.uint8)

print("Difference filtered image shape:",img_diff.shape)

plt.figure(4)
plt.imshow(img_diff)
img_diff = cv.cvtColor(img_diff,cv.COLOR_RGB2BGR)
cv.imwrite("difference_barb.jpeg",img_diff)


#6 horizontal edges
horizontal_filter = np.array([[-1,2,-1]])


horizR= convolve2d(imgR,horizontal_filter,mode="same")
horizG= convolve2d(imgG,horizontal_filter,mode="same")
horizB= convolve2d(imgB,horizontal_filter,mode="same")
img_horiz = np.dstack( (horizR,horizG,horizB) ).astype(np.uint8)

print("Horizontal Image filtered shape:",img_horiz.shape)

plt.figure(5)
plt.imshow(img_horiz)
img_horiz = cv.cvtColor(img_horiz,cv.COLOR_RGB2BGR)
cv.imwrite("Horizontal_barb.jpeg",img_horiz)

#7 vertical edges

vertical_filter = horizontal_filter.transpose()


vertR= convolve2d(imgR,vertical_filter,mode="same")
vertG= convolve2d(imgG,vertical_filter,mode="same")
vertB= convolve2d(imgB,vertical_filter,mode="same")
img_vert = np.dstack( (vertR,vertG,vertB) ).astype(np.uint8)

print("Vertical Image filtered shape:",img_horiz.shape)

plt.figure(6)
plt.imshow(img_vert)
img_vert = cv.cvtColor(img_vert,cv.COLOR_RGB2BGR)
cv.imwrite("vert_barb.jpeg",img_vert)


plt.show()