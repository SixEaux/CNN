import numpy as np

class Dense:
    """Dense layer.

        Args:
            number_neurons (int): number of layers
            learning_rate (float): learning rate
        """
    def __init__(self, number_neurons:int, learning_rate:float):
        self.weight = None # weights of (number_neurons, length_input)
        self.bias = None # bias of (1, number neurons)
        self.input = None # input to layer
        self.learning_rate = learning_rate
        self.number_neurons = number_neurons

    def initial_param(self, dim_in:int):
        """Initialize the parameters.

        Args:
            dim_in (int): dimensions entering the layer.
        """
        self.weight = np.random.uniform(-1, 1, (self.number_neurons, dim_in)).astype(np.float32)
        self.bias = np.zeros((1, self.number_neurons)).astype(np.float32)

    def forward(self, x:np.ndarray):
        """Forward propagation dense layer.

        Args:
            x (ndarray): input to the layer. DIM = (batch_size, shape_input_flattened)

        Returns:
            ndrray: output of the layer. DIM = (batch_size, number_neurons)
        """
        self.input = x
        return x @ self.weight.T + self.bias #bias broadcasted across batch
    
    def backward(self, dL_dout:np.ndarray, batch_size:int=1):
        """Recover gradient layer before and actualise weights.

        Args:
            dL_dout (ndarray): gradient next layer. DIM = (batch_size, number_neurons) 

        Returns:
            ndarray: gradient for layer before. DIM = (batch_size, length_input)
        """

        dC_dw = dL_dout.T @ self.input # (number_neurons, length_input) 
        dC_db = np.sum(dL_dout, axis=0, keepdims=True) # here i need a sum across batches

        dL_dout = dL_dout @ self.weight #gradient for layer before
        
        # actualise weights and bias
        self.weight -= self.learning_rate * dC_dw / batch_size
        self.bias -= self.learning_rate * dC_db / batch_size

        return dL_dout