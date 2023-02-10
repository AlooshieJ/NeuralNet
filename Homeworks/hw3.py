import pandas as pd
import numpy as np
import torch

"""
ECE 491 - intro to neural networs Homework 3
Programming linear predictor, calculating avg. error
Mean Squared Error: MSE =
(1/N) * (SUM) i = 1 -> n ( x^ [n] - x[n])^2 
x^ is predicted  value, x[n] is true value 
"""

# 2nd order adaptive filter NO Bias
class Predictor:
    def __init__(self,order=2, bias =0 ):
        # self.Data = data
        # self.n = len(data)
        self.bias = bias
        self.order= order


    # prediction for 2nd order uses x[n], x[n-1] to predict x[n+1]
    # X = [x1,x2] = [ x[n] , x[n-1] ]
    #returns output y[n] = dot product of X w/ weight
    def predict(self,X,W):
        #print(X,W)
        return np.dot(X,W) + self.bias

    # return error squared
    def error_calc(self,predicted,actual):
        #print(predicted,actual)
        return (actual-predicted)

    # 2nd order MSE
    def MSE(self,xn,w1):

        error_sum = 0
        w = w1
        error = 0
        print("Input Data: ",xn,"\nInitial weight: ",w1,"\n-----------------")

        for i in range(len(xn)):
            #predict
            X = xn[i:(i+1)+1]
            prediction = self.predict(X,w) + self.bias
            try:
                actual = xn[i+2]
            except IndexError:
                print("final Prediction:",prediction)
                break
            error = self.error_calc(prediction,actual)
            error_sum += (error**2)

            #updating weights
            f = f"X: {X} prediciton: {prediction} | {actual}, w = {w},error:{error}"
            print(f,end=" ")
            n = 1/ (np.linalg.norm(X))**2
            w = w +  n * np.dot(error,X)
            print("updated w :",w)
            print("----------------")

        mse = 1/len(xn) * error_sum
        f = f"avg. error (MSE) = {mse}"
        print(f)



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
P = Predictor(bias = 1)
#P.MSE(DataHW,[.5,.5])
P.MSE(inputs,[1,1])
#print(inputs)
