import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn



class NeuralNet():
    def __init__(self,w,b):
        super(NeuralNet,self).__init__()
        self.Weights = w
        self.Bias = b
        # self.linear.weight = weight
        # self.linear.bias = bias

    def activate_step(self,n):
        if n>=0:
            return 1
        else:
            return 0

    # guess
    def forward(self,x):
        #return torch.dot(x,self.Weights) + self.Bias
        res = torch.dot(self.Weights,x) + self.Bias
        return self.activate_step(res)

    def run_epoch(self,X,y,epochs=10):
        print(f"Initial W = {self.Weights}")

        for e in range(epochs):
            print(f"epoch #{e}\n------------")
            self.Weights = self.train_loop(X,y)

    def train_loop(self,X,label):
        w= self.Weights
        for index,x in enumerate(X):
            y = label[index]
            pred=self.forward(x)

            error = y - pred
            print(f"x:{x} | guess | {pred} | y: {y} | error: {error}")
            #calculating the new weights
            if pred != y:
                w += error*x

        return w

    def get_w_b(self):
        return(self.Weights,self.Bias)
# we are given this data set
# C0: As shown in orange, [1, -1], [0, 0], [-1, 1], with label 0;
# C1: As shown in blue, [3, 2], [2, 2], [2, 3], with label 1;
Data = [
    [1,-1], # label 0
    [0,0],
    [-1,1],
    [3,2],# label 1
    [2,2],
    [2,3],
        ]
labels = np.hstack([np.zeros(shape=3),np.ones(shape=3)])
X = torch.tensor(Data,dtype=torch.float32)
x_labels=torch.tensor(labels)
#initial weights and bias
W = torch.tensor([1.,1.])
bias = 1


# draw the initial graph
x_axis_points = torch.tensor(np.arange(-1,4,1))
fig = plt.figure(1)
plt.scatter(X[:,0],X[:,1] , c=labels)
slope = W[1] / W[0]
y = x_axis_points *slope + bias
plt.plot(x_axis_points,y,label="Initial Line")

epoch = 20
model = NeuralNet(W,bias)
model.run_epoch(X,labels,epoch)
#print(model.get_w_b())

# drawing the final line
final_W= model.get_w_b()[0]
print(f"final Weight: {final_W}")
slope = final_W[1]/ final_W[0]
y_new = x_axis_points * slope

plt.plot(x_axis_points,y_new,label="Final Line")
# P = Perceptron(X,labels,bias,debug=True)
# final_w = P.train(torch.tensor(W),5)

plt.legend()
plt.show()