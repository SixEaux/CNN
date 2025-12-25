import numpy as np

class Loss:
    """
    Loss "layer". Might in the future delete class and add it directly to NN.

        Args:
            function (str): what error function to use: MSE or CE (CE directly uses softmax).
    """

    def __init__(self, function):
        self.function = function
        self.expected = None
        self.observed = None
    
    def initial_param(self, dim_in): # just so every layer has one
        return

    def softmax(self, x):
        """Softmax function.

        Args:
            x (ndarray): input. DIM = (number_neurons_last_layer, 1)

        Returns:
            ndarray: output. DIM = (number_neurons_last_layer, 1)
        """
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return  exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def one_hot_vector(self, expected):
        """Transform to one hot vector (vector with one where it was expected)

        Args:
            expected (int or list): expected output

        Returns:
            ndarray: DIM = (number_classes,)
        """
        
        return np.eye(10)[expected].T
    
    def forward(self, obs, exp, nbinput=1):
        """Calculate loss.

        Args:
            obs (ndarray): observed output. DIM = (number_calsses, 1)
            exp (ndarray): expected output. DIM = (number_calsses, 1)
            nbinput (int): batch size

        Returns:
            float: loss value of size batch (it can be divised by nbinput to have the mean but not necessary)
        """

        exp = self.one_hot_vector(exp)

        if self.function == "CEL": # does the softmax also directly
            self.observed = self.softmax(obs)
            self.expected = exp
            loss = -np.sum(exp * np.log(self.observed + 1e-12), axis=0) 
            return loss
        elif self.function == "MSEL":
            self.observed = obs
            self.expected = exp
            loss = (np.sum((obs - exp) ** 2, axis=0))/ 2
            return loss
        else:
            raise ValueError("Not a valid loss function.")

    def backward(self, nbinput=1):
        """Recover gradient for layer before.

        Args:
            nbinput (int): batch size

        Raises:
            ValueError: if the loss function unknown

        Returns:
            ndarray: gradient wrt output layer before (z if CEL or a if MSEL). DIM = (number_classes,)
        """
        if self.function == "CEL":
            return (self.observed - self.expected) / nbinput
        elif self.function == "MSEL":
            return (self.observed - self.expected) / nbinput
        else:
            raise ValueError("Not a valid loss function.")