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
n = np.arange(-40, 80, 1)
# x[n] function
x = 7 * np.cos(0.1 * n) + np.cos(0.95 * n)
# k = np.arange(-40,80,1)
# xofn =[7 * np.cos(0.1 * n) + np.cos(0.95 * n) for n in np.arange(-40,80,1)]
fig5 = plt.figure(5)
plt.stem(n, x)

#x[n-20]
x = 7 * np.cos(0.1 * n-20) + np.cos(0.95 * n-20)
fig6 = plt.figure(6)
plt.stem(n, x)

"""
PART B)
"""
# length of array
N = 20
h = 1./5. * np.array([1*N])
k = np.arange(0,20,1)
#2a
y=np.convolve(np.ones(shape=N),h)
print(y)
fig7 = plt.figure(7)
plt.stem(k,y)

#2b
k = np.arange(-40,80,1)
y = np.convolve(np.cos(0.1*k),h)
fig8=plt.figure(8)
plt.stem(k,y)
print(y)
plt.show()