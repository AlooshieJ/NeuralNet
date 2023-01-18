import torch
import d2l
import numpy as np
import matplotlib.pyplot as plt

"""
x = torch.arange(12,dtype=torch.float32)
print(x)

print(
x.numel(),"\n", # elements in x
x.shape) # length along each axis

X =x.reshape(3,4) # creates a matrix with x elements

print(torch.exp(x)) # exponent


# ------ MATRIX -------
A = torch.arange(6).reshape(3, 2)
print (A, "\n",A.T) # A.T is transpose
"""

# equation 2x1 + 3x2 + 4x3 - 4 = 0
# let x1 = x , x2 = y , x3 = z... then:
#z = -1/2 * x - 3/4 * y + 1
#
# create plot figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# setup data for plotting
xx = np.arange(-10,10,1,dtype="float32")
yy = np.arange(-10,10,1,dtype="float32")

X,Y = np.meshgrid(xx,yy)

# calculate the z / for plotting
Z = -1*(1/2) * (X) - (3/4)*Y + 1
#print(Z)
ax.plot_surface(X,Y,Z)

#axis label / customize
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
