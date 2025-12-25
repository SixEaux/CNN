import numpy as np
from scipy.special import expit


class Activation:
    """Activation layer.

        Args:
            function (str): which activation function is used from: 
            sigmoid, relu, tanh.
        """

    def __init__(self, function):
        self.function = function
        self.input = None

    def initial_param(self, dim_in): # just so every layer has one
        return

    def forward(self, x):
        """Apply activation function to input.

        Args:
            x (ndarray): input from last layer

        Raises:
            ValueError: if the activation function is not known

        Returns:
            ndarray: activated output
        """
        if self.function == "sigmoid":
            self.input = x
            a = expit(x)
            return a
        elif self.function == "relu":
            self.input = x
            a = np.maximum(x, 0)
            return a
        elif self.function == "tanh":
            self.input = x
            a = np.tanh(x)
            return a
        else:
            raise ValueError("Activation function is not valid.")
    
    def backward(self, dL):
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
        
