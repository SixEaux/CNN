import numpy as np
from scipy.special import expit
from src.layer import Layer

class Activation(Layer):
    """Activation layer.

        Args:
            function (str): which activation function is used from: 
            sigmoid, relu, tanh.
        """

    def __init__(self, function:str):
        self.function = function
        self.input = None 
        self.output = None #keep track of output not to recompute in sigmoid/tanh

    def initial_param(self, *args): # just so every layer has one
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
            self.output = expit(x)
            return self.output
        elif self.function == "relu":
            self.input = x
            return np.maximum(x, 0)  
        elif self.function == "tanh":
            self.input = x
            self.output = np.tanh(x)
            return self.output
        else:
            raise ValueError("Activation function is not valid.")
    
    def backward(self, dL:np.ndarray, *args):
        """Recover gradient for backward propagation.

        Args:
            dL (ndarray): gradient from next layer

        Raises:
            ValueError: if the activation function is not known

        Returns:
            ndarray: gradient wrt loss by chain rule : (d_C / d_a) * (d_a / d_z_layerbefore)
        """

        if self.function == "sigmoid":
            return dL * self.output*(1-self.output) 
        elif self.function == "relu":
            return dL * np.where(self.input>=0, 1, 0)
        elif self.function == "tanh":
            return dL * (1 - self.output**2)
        else:
            raise ValueError("Activation function is not valid.")
        
