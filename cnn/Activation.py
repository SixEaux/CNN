import numpy as np
from scipy.special import expit

class Activation:
    """Activation layer.

        Args:
            function (str): which activation function is used from: 
            sigmoid, relu, tanh.
        """

    def __init__(self, function:str):
        self.function = function
        self.input = None

    def initial_param(self, dim_in): # just so every layer has one
        return

    def forward(self, x:np.ndarray):
        """Apply activation function to input.

        Args:
            x (ndarray): input from last layer. DIM = input_shape

        Raises:
            ValueError: if the activation function is not known

        Returns:
            ndarray: activated output. DIM = input_shape
        """
        if self.function == "sigmoid":
            self.input = x
            return expit(x)
        elif self.function == "relu":
            self.input = x
            return np.maximum(x, 0)  
        elif self.function == "tanh":
            self.input = x
            return np.tanh(x)
        else:
            raise ValueError("Activation function is not valid.")
    
    def backward(self, dL:np.ndarray, batch_size=1):
        """Recover gradient for backward propagation.

        Args:
            dL (ndarray): gradient from next layer

        Raises:
            ValueError: if the activation function is not known

        Returns:
            ndarray: gradient wrt loss by chain rule : (d_C / d_a) * (d_a / d_z_layerbefore)
        """

        if self.function == "sigmoid":
            return dL * expit(self.input)*(1-expit(self.input)) 
        elif self.function == "relu":
            return dL * np.where(self.input>=0, 1, 0)
        elif self.function == "tanh":
            return dL * (1 - np.tanh(self.input)**2)
        else:
            raise ValueError("Activation function is not valid.")
        
