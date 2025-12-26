import numpy as np

class Dense:
    """Dense layer.

        Args:
            number_neurons (int): number of layers
            learning_rate (float): learning rate
        """
    def __init__(self, number_neurons:int, learning_rate:float):
        self.w = None # weights of (number neurons, number inputs)
        self.b = None # biais of (number neurons, 1)
        self.input = None # input to layer
        self.learning_rate = learning_rate
        self.number_neurons = number_neurons

    def initial_param(self, dim_in):
        """Initialize the parameters.

        Args:
            dim_in (int): dimensions entering the layer.
        """
        self.w = np.random.uniform(-1, 1, (self.number_neurons, dim_in))
        self.b = np.random.uniform(-1, 1, (self.number_neurons,1))

    def forward(self, x):
        """Forward propagation dense layer.

        Args:
            x (ndarray): input to the layer. DIM = (batch_size, shape_input_flattened)

        Returns:
            ndrray: output of the layer. DIM = (number_neurons, 1)
        """
        self.input = x
        return self.w @ x + self.b
    
    def backward(self, dL_dout):
        """Recover gradient layer before and actualise weights.

        Args:
            dL_dout (ndarray): gradient next layer. DIM = (number_neurons, 1) 

        Returns:
            ndarray: gradient for layer before. DIM = (number_inputs, 1)
        """
        dC_dw = np.outer(dL_dout, self.input) # same as np.dot(dL_dout, self.input.T) # (number_neurons, length_input)
        dC_db = dL_dout # here i need a sum for batch i think
        print(dC_db.shape)

        dL_dout = self.w.T @ dL_dout #gradient for layer before
        
        # actualise weights and bias
        self.w -= self.learning_rate * dC_dw
        self.b -= self.learning_rate * dC_db

        return dL_dout