import numpy as np
import matplotlib.pyplot as plt

"""
----------------
ECE 491 : Intro to Neural Networks 
LAB #1 - Intro to Python and 1-D convolution
By: Ali Jafar 
UIN: 669430206
----------------
"""

"""
PART A1) 
"""
# creating vector x with numpy specifying the type of: float32
# by using keyword dtype
x = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
fig1 = plt.figure(1)
plt.stem(x)
plt.pause(0.0001)
# creating a new vector (k) of range [-2 : 7]
k = np.arange(-2, 8, 1, dtype=np.float32)
#plotting k
fig2 = plt.figure(2)
plt.stem(k, x)

"""
Part A2)
"""
N, M = 10, 20
x = np.hstack([np.zeros(shape=N), np.ones(shape=(M+1))])
#print(x)
#k is range of [-10 : 20]
k = np.arange(-10, 21, 1)
# u[n]
fig3 = plt.figure(3)
plt.stem(k, x)

#u[n-5]
N, M = 15, 15
x = np.hstack([np.zeros(shape=N), np.ones(shape=(M+1))])
fig4 = plt.figure(4)
plt.stem(k, x)

"""
PART A3)
"""
k = np.arange(-40, 80, 1)

# x[n] function
n = np.arange(0,len(k),1)
x = 7 * np.cos(0.1 * n) + np.cos(0.95 * n)
fig5 = plt.figure(5)
plt.stem(k, x)

#x[n-20]
x = 7 * np.cos(0.1 * n-20) + np.cos(0.95 * n-20)
fig6 = plt.figure(6)
plt.stem(k, x)

"""
PART B)
"""
N=5
# length of array
h=1./5.*np.array([1.,1.,1.,1.,1.])
k = np.arange(0,21,1)

#NOTE: The keyword mode = "same" for the convolution this keeps the output length in the boundry
# of the inputs given
#2a
x= np.ones(shape=len(k))
y=np.convolve(x, h,mode="same")
fig7 = plt.figure(7)
plt.stem(k, y)

k = np.arange(-40,80,1)
#2b
x = np.cos(0.1*k)
y = np.convolve(x,h,mode="same")
fig8=plt.figure(8)
plt.stem(k,y)

#2c
x = np.cos(0.95*k)
y = np.convolve(x,h,mode="same")
fig9=plt.figure(9)
plt.stem(k,y)

#2d
n = np.arange(0,len(k),1)
x = 7 * np.cos(0.1 * n) + np.cos(0.95 * n)
y = np.convolve(x,h,mode="same")
fig10=plt.figure(10)
plt.stem(k,y)


#3a)
h = [1,-1]
k = np.arange(0,21,1)
x = np.ones(shape= len(k))
y = np.convolve(x,h,mode="same")
fig11=plt.figure(11)
plt.stem(k,y)

#3b
k = np.arange(-40,80,1) #new k range
x = np.cos(0.1*k)
y = np.convolve(x,h,mode="same")
fig12 = plt.figure(12)
plt.stem(k,y)

#3c
x = np.cos(0.95*k)
y = np.convolve(x,h,mode="same")
fig13 = plt.figure(13)
plt.stem(k,y)

#3d
n = np.arange(0,len(k),1)
x = 7 * np.cos(0.1 * n) + np.cos(0.95 * n)
y = np.convolve(x,h,mode="same")
fig14=plt.figure(14)
plt.stem(k,y)

plt.show()
