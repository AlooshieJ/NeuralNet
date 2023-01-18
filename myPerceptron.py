import torch
import d2l
import numpy as np


# from d2l book This function intended for jupyter notebook programming.
def add_to_class(Class):  #@save
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


# percpetron class, will have inputs:
# ( X,y) where X is an n-dim vector
# y = {-1, 1} depending on classifier
#

class Perceptron:

    #The __init__ method stores the learnable parameters,

    def __init__(self, Training_set):
        self.Data = Training_set
        self. y  = []

        pass
    # the train- ing_step method accepts a data batch to return the loss value,
    def training_step(self):
        pass

    #the configure_optimizers method returns the optimization method,
    # or a list of them, that is used to update the learnable parameters.
    def config_optimizers(self):
        pass


