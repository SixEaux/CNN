import numpy as np
from src.parameter_initialization import he_initialization, xavier_initialization
from src.layer import Layer


class Dense(Layer):
    """Dense layer.

    Args:
        number_neurons (int): number of layers
        initialization (str): what type of initialization between: he, xavier
    """

    def __init__(self, number_neurons: int, initialization: str = "xavier"):
        self.weight = None  # weights of (number_neurons, length_input)
        self.bias = None  # bias of (1, number neurons)
        self.input = None  # input to layer
        self.number_neurons = number_neurons
        self.initialization = initialization

        self.dW = None  # gradient for weights
        self.dB = None  # gradient for bias

    def initial_param(self, dim_in: int, dim_out: int):
        """Initialize the parameters.

        Args:
            dim_in (int): dimensions entering the layer.
        """
        if self.initialization == "he":
            self.weight = (
                he_initialization(dim_in, self.number_neurons * dim_in)
                .reshape((self.number_neurons, dim_in))
                .astype(np.float32)
            )
        elif self.initialization == "xavier":
            self.weight = (
                xavier_initialization(dim_in, dim_out, self.number_neurons * dim_in)
                .reshape((self.number_neurons, dim_in))
                .astype(np.float32)
            )
        elif self.initialization == "random":
            self.weight = np.random.uniform(-1, 1, (self.number_neurons, dim_in)).astype(np.float32)
        else:
            raise ValueError("I don't know this initialization.")

        self.bias = np.zeros((1, self.number_neurons)).astype(np.float32)

        self.out_dim = self.number_neurons

    def forward(self, x: np.ndarray):
        """Forward propagation dense layer.

        Args:
            x (ndarray): input to the layer. DIM = (batch_size, shape_input_flattened)

        Returns:
            ndrray: output of the layer. DIM = (batch_size, number_neurons)
        """
        self.input = x
        return x @ self.weight.T + self.bias  # bias broadcasted across batch

    def backward(self, dL_dout: np.ndarray, batch_size: int = 1, just_computation: bool = False):
        """Recover gradient layer before and actualise weights.

        Args:
            dL_dout (ndarray): gradient next layer. DIM = (batch_size, number_neurons)
            batch_size (int): size of the batch
            just_computation (bool): whether to just compute the gradients without updating weights

        Returns:
            ndarray: gradient for layer before. DIM = (batch_size, length_input)
        """
        dC_dw = dL_dout.T @ self.input  # (number_neurons, length_input)
        dC_db = np.sum(dL_dout, axis=0, keepdims=True)  # here i need a sum across batches

        dL_dout = dL_dout @ self.weight  # gradient for layer before

        if not just_computation:
            self.dW = dC_dw / batch_size
            self.dB = dC_db / batch_size

            return dL_dout
        else:
            return dC_dw / batch_size, dC_db / batch_size