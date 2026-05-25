import numpy as np
from src.layer import Layer
from functools import reduce 

class Flattening(Layer):
    def __init__(self):
        self.input = None
        self.out_dim = None
    
    def initial_param(self, *args): # just so every layer has one
        self.out_dim = reduce(lambda x, y: x * y, args[0])

    def forward(self, x:np.ndarray):
        """Reshape input for fully connected layer.

        Args:
            x (ndarray): input (batch_size, h, w, c)

        Returns:
            ndarray: reshaped to (batch_size, h*w*c)
        """
        self.input = x
        return x.reshape((x.shape[0], -1))
    
    def backward(self, dL_dout:np.ndarray, *args):
        """Reshape for backpropagation.

        Args:
            dL_dout (np.ndarray): gradient error from next layer
            batch_size (int, optional): size of the batch. Defaults to 1.

        Returns:
            np.ndarray: gradient error with same shape as input
        """
        return dL_dout.reshape(self.input.shape)