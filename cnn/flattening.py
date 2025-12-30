import numpy as np

class Flattening:
    def __init__(self):
        self.input = None
    
    def initial_param(self, dim_in): # just so every layer has one
        return

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