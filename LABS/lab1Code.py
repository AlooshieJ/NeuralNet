import torch
import numpy as np
import matplotlib.pyplot as plt

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
# k = np.arange(-40,80,1)
# xofn =[7 * np.cos(0.1 * n) + np.cos(0.95 * n) for n in np.arange(-40,80,1)]
fig5 = plt.figure(5)
plt.stem(k, x)

#x[n-20]

x = 7 * np.cos(0.1 * n-20) + np.cos(0.95 * n-20)
fig6 = plt.figure(6)
plt.stem(k, x)

"""
PART B)
"""
# length of array
N = 5
h = 1./5. * np.array([1*N])
k = np.arange(0,20,1)

#2a

x= np.ones(shape=len(k))
y=np.convolve(x, h)
fig7 = plt.figure(7)
plt.stem(k, y)

k = np.arange(-40,80,1)
#2b
x = np.cos(0.1*k)
y = np.convolve(x,h)
fig8=plt.figure(8)
plt.stem(k,y)

#2c
x = np.cos(0.95*k)
y = np.convolve(x,h)
fig9=plt.figure(9)
plt.stem(k,y)

#2d
n = np.arange(0,len(k),1)
x = 7 * np.cos(0.1 * n) + np.cos(0.95 * n)
y = np.convolve(x,h)
fig10=plt.figure(10)
plt.stem(k,y)

plt.show()
