import torch
from d2l import torch as d2l
import numpy as np
import matplotlib.pyplot as plt



# from d2l book This function adds atrributes to class.
def add_to_class(Class):  #@save
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper



# perceptron implementation from scratch
class Perceptron():
    def __init__(self,bigX ,y,bias,debug = False):
        """

        :param bigX: The n dim. Data to train with; features
        :param y: The class for each data point
        """
        self.X = bigX # matrix of features
        self.y  = y # matrix of class for each data point

        self. n = len(self.X[0]) # dimensions of feature matrix
        self.b = bias

        self.debug = debug

    def showdata2d(self):
        print(f"-----Dimensions of data -----")
        print(f"n = {self.n}, bias = {self.b}")
        print(f"X shape = {self.X.shape} | y shape = {self.y.shape}\n")
        index = 0
        for x1,x2 in self.X:
            print(f"({x1},{x2}) -> class: {self.y[index]}")
            index +=1

    # perceptron activation function
    def activate(self, n):
        if n >= 0:
            return 1
        else:
            return -1
# runs the perceptron algorithm
    def forward(self,weights):
        # initialize the w vector
        w = weights
        b = self.b
        index = 0
        if self.debug:
            print(f"Initialize W = {w}")
        #loop through all the data and perform prediction
        for x in self.X:
            y = self.y[index]
            # predict
            y_hat = torch.dot(x,w) + b

            #activate function returns sign of dot product result
            y_hat = self.activate(y_hat)
            if self.debug:
                print(f"index: {index}, {x}DOT{w} = y_hat: {y_hat} | {y}")

            #update based on result of activation function
            if(y_hat != y):
                w = w + (x*y)

            index +=1
        return w

    #train function runs the percpetron algorithm with activation function
    #to update the weight vector based on previous data
    def train(self,weights ,iterations=50):
        w_train = weights
        for n in range(iterations):
            print(f"--TRINAING-- iter: {n}")
            w_train = self.forward(w_train)
            print(f"updated w = {w_train}")
        return w_train


#runing the perceptron class
# in form [x1 x2]
#preparing data from graph in homework
reds= torch.tensor([[1,1],[2,1],[1,2],[2,2]],dtype=torch.float32)# = 1
blues = torch.tensor([[0,-1], [-1,-1],[-1,0]],dtype=torch.float32) # = -1

data_class = torch.tensor([1,1,1,1,-1,-1,-1],dtype=torch.float32)
data= torch.cat((reds,blues),0)

P = Perceptron(data, data_class,1)

# starting weight vectors, using the ones for HW Q
#inital_weights = torch.zeros(2,dtype=torch.float32)
inital_weights = torch.ones(2,dtype=torch.float32)

final_w = P.train(inital_weights,iterations= 5)

print( "Final W  = " , final_w ,f"with Bias of {P.b}" )

# plt.scatter(data[:,0],data[:,1])
# plt.show()
