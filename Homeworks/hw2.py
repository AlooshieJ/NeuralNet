import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


class NN (nn.Module):
    def __init__(self,lr=0.01):
        super(NN,self).__init__()
        self.net =  nn.Linear(2,1)
        self.net.weight.data.fill_(1.) # initializing weight
        self.net.bias.data.fill_(1) # initialize bias
        print(self.net.weight.data,self.net.bias.data)
        self.lr = lr #learning rate

    def forward(self,X):
        return self.net(X)


    def loss(self,y_hat,y):
        #print(y_hat,y_hat.size(),y,y.size())
        fn= nn.MSELoss()
        return fn(y_hat,y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),self.lr)

    def train_loop(self,data,labels):
        size = len(data)
        for index, X in enumerate(data):

            y = labels[index]

            #compute prediciton and loss
            pred = self.net(X)
            y=y.reshape_as(pred)

            loss = self.loss(pred,y)

            #backprop
            self.configure_optimizers().zero_grad()
            loss.backward()
            self.configure_optimizers().step()

            loss,current = loss.item(), index +1
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def get_w_b(self):
        return(self.net.weight.data,self.net.bias.data)

# we are given this data set
# C0: As shown in orange, [1, -1], [0, 0], [-1, 1], with label 0;
# C1: As shown in blue, [3, 2], [2, 2], [2, 3], with label 1;
points = [
    [1,-1], # label 0
    [0,0],
    [-1,1],
    [3,2],# label 1
    [2,2],
    [2,3],
        ]
labels = np.hstack([np.zeros(shape=3),np.ones(shape=3)])
X = torch.tensor(points,dtype=torch.float32,requires_grad=True)
x_labels=torch.tensor(labels,dtype=torch.float32,requires_grad=True)

#initial weights and bias
W = torch.tensor([1.,1.])
bias = 1


# draw the initial graph & line
x_axis_points = torch.tensor(np.arange(-1,4,1))
fig = plt.figure(1)
plt.scatter(X[:,0].detach().numpy(),X[:,1].detach().numpy() , c=labels)
slope = W[1] / W[0]
y = x_axis_points *slope + bias
plt.plot(x_axis_points,y,label="Initial Line")


model = NN(lr = 0.01)

#running the model
epochs = 20
for e in range(epochs):
    print(f"Epoch {e}\n---------------")
    model.train_loop(X,x_labels)
print("DONE!")
#print(model.get_w_b())


# drawing the final line
final_W= model.get_w_b()[0][0]
final_b = model.get_w_b()[1]

print(f"final Weight: {final_W} final bias: {final_b}")
slope = -1*final_W[1]/ final_W[0]
y_new = x_axis_points * slope +final_b
plt.plot(x_axis_points,y_new,label="Final Line")

plt.legend()
plt.show()