import numpy as np

class Loss:
    """
    Loss "layer".

        Args:
            function (str): what error function to use: MSE or CE (CE directly uses softmax).
    """

    def __init__(self, function:str):
        self.function = function
        self.expected = None
        self.observed = None
    
    def initial_param(self, dim_in): # just so every layer has one
        return

    def softmax(self, x:np.ndarray):
        """Softmax function.

        Args:
            x (ndarray): input. DIM = input_shape (normally (batch_size, number_neurons_last_layer))

        Returns:
            ndarray: output. DIM = input_shape (normally (batch_size, number_neurons_last_layer))
        """

        #the sum and the max are internally in each sample

        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return  exp_x / np.sum(exp_x, axis=1, keepdims=True) 

    def one_hot_vector(self, expected:np.ndarray):
        """Transform to one hot vector (vector with one where it was expected)

        Args:
            expected (ndarray): expected output. DIM = (batch_size,)

        Returns:
            ndarray: DIM = (batch_size, number_classes)
        """

        return np.eye(10)[expected.squeeze()]
    
    def forward(self, obs:np.ndarray, exp:np.ndarray):
        """Calculate loss.

        Args:
            obs (ndarray): observed output. DIM = (batch_size, number_classes)
            exp (ndarray): expected output. DIM = (batch_size,)
            batch_size (int): batch size

        Returns:
            ndarray: loss value. Dim = (batch_size, 1)
        """

        exp = self.one_hot_vector(exp) # (batch_size, number_classes)

        if self.function == "CEL": # does the softmax also directly
            self.observed = self.softmax(obs)
            self.expected = exp
            loss = -np.sum(exp * np.log(self.observed + 1e-12), axis=1, keepdims=True) 
            return loss
        elif self.function == "MSEL":
            self.observed = obs
            self.expected = exp
            loss = (np.sum((obs - exp) ** 2, axis=1, keepdims=True)) / 2
            return loss
        else:
            raise ValueError("Not a valid loss function.")

    def backward(self):
        """Recover gradient for layer before.

        Args:
            

        Raises:
            ValueError: if the loss function unknown

        Returns:
            ndarray: gradient wrt output layer before (z if CEL or a if MSEL). DIM = (batch_size, number_classes)
        """

        if self.function == "CEL":
            return (self.observed - self.expected)
        elif self.function == "MSEL":
            return (self.observed - self.expected)
        else:
            raise ValueError("Not a valid loss function.")