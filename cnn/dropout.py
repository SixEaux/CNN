import numpy as np
from cnn.layer import Layer

class Dropout(Layer):
    def __init__(self, drop_rate:float):
        """Dropout layer. Deactivates some neurons to prevent overfitting /
        relying only in some neurons.
        I don't know if it is really useful but found it interesting and wanted to try it.

        Args:
            drop_rate (float): part to drop
        """
        self.drop_rate = drop_rate
        self.generator = np.random.default_rng()
        self.mask = None

    def initial_param(self, *args): # just so every layer has one
        return
    
    def forward(self, x:np.ndarray):
        """Forward pass. Set some neurons to zero.

        Args:
            x (np.ndarray): input

        Returns:
            np.ndarray: input masked
        """
        self.mask = self.generator.binomial(1, 1 - self.drop_rate, x.shape)
        return x * self.mask / (1-self.drop_rate)

    def backward(self, dL_dout:np.ndarray, *args):
        """Backward pass. Just return the gradient that arrives.

        Args:
            dL_dout (np.ndarray): gradient from next layer

        Returns:
            np.ndarray: dL_dout
        """
        return dL_dout * self.mask / (1 - self.drop_rate)
    