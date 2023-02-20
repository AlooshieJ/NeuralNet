import pandas as pd
import numpy as np
import torch

"""
ECE 491 - intro to neural networs Homework 3
Programming linear predictor, calculating avg. error
Mean Squared Error: MSE =
(1/N) * (SUM) i = 1 -> n ( x^ [n] - x[n])^2 
"""

# n-order adaptive filter predictor class
class Predictor:
    def __init__(self,order=2, bias =0 , debug = False ):
        self.bias = bias
        self.order= order
        self.debug = debug


    # prediction for 2nd order uses x[n], x[n-1] to predict x[n+1]
    # X = [x1,x2] = [ x[n] , x[n-1] ]
    #returns output y[n] = dot product of X w/ weight
    def predict(self,X,W):
        return np.dot(X,W)

    # return error squared
    def error_calc(self,predicted,actual):
        #print(predicted,actual)
        return (actual-predicted)

    # 2nd order MSE
    def MSE(self,xn,w1):

        error_sum = 0
        w = w1
        error = 0
        count = 0
        if self.debug:
            print("Input Data: ",xn,"\nInitial weight: ",w1,"\n-----------------")

        for i in range(len(xn)):
            if self.debug:
                print("iteration: ",i)
            #predict
            #X = xn[i:(i+1)+1]
            X = xn[i:i+self.order]
            X = np.flip(X)
            X = np.append(X,self.bias)
            prediction = self.predict(X,w)

            # condition to get out of predictor loop
            try:
                actual = xn[i+self.order]
            except IndexError:
                print("final Prediction:",prediction,"Final Weights: ",w,"Bias: ",w[self.order])
                break

            error = self.error_calc(prediction,actual)
            error_sum += error**2
            count += 1

            if self.debug:
                f = f"X: {X} prediciton: {prediction} | {actual}, w = {w},error:{error}"
                print(f,end=" ")

            #updating weights
            n = 1/ (np.linalg.norm(X))**2
            w = w + n * error * X#np.dot(error,X)
            if self.debug:
                print("updated w :",w)
                print("----------------")
        mse = (1/count) * error_sum
        #mse /= 100
        # divide by 100 for the percent
        f = f"avg. error (MSE) = {mse}%"
        print(f)
        print("-----------------")



# parsing data down to array of type float
#note, data given in order from most recent to latest
Data = pd.read_csv('HistoricalQuotes.csv')
DataHW = [1070,2202,3010,4090,4900]
inputs = []
for a in Data['Close/Last']:
    #print(a)
    inputs.append(float(a.replace('$','')))

#reversing data, so that the earliest value is first
#print(inputs)
inputs.reverse()
inputs = np.array(inputs)
# testing w/ q1 homework data
#P = Predictor()
#P.MSE(DataHW,[.5,.5])


# # q 2a 2nd order predictor no bias
Pa = Predictor()
Pa.MSE(inputs,np.array([.5,.5,0]))


# q 2b 2nd order predictor with bias
Pb = Predictor(bias=-1)
Pb.MSE(inputs,[.5,.5,10])
# # q 2c 3rd order predictor no bias
Pc = Predictor(order = 3)
Pc.MSE(inputs,[.5,.5,.5,0])
# # q 2d 3rd order predictor with bias
Pd = Predictor(order= 3,bias =-1)
Pd.MSE(inputs,[.5,.5,.5,15])
