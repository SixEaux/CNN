# TODO : changes for batch

import numpy as np

class Model:
    """Model NN.

        Args:
            layers (list[Layers]): list of layers (objects)
            loss (Loss): loss object 
            dataset (str): dataset used
        """
    def __init__(self, layers:list, loss, dataset:str):
        self.layers = layers
        self.loss = loss
        self.dataset = dataset

        self.model_initial()

    def model_initial(self):
        """Initialize the model based on dimensions input.
        """
        dims_dataset = {"mnist":(28,28,1)}
        last_dim_out = dims_dataset[self.dataset]
        last_dim_out = last_dim_out[0] * last_dim_out[1]

        for l in self.layers:
            l.initial_param(last_dim_out)
            last_dim_out = l.number_neurons if hasattr(l, "number_neurons") else last_dim_out

    def softmax(self, x):
        """Softmax function.

        Args:
            x (ndarray): input. DIM = (number_neurons_last_layer,)

        Returns:
            ndarray: output. DIM = (number_neurons_last_layer,)
        """
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return  exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, x, expected):
        #TODO : see later for minibatch
        """Forward propagation through all the layers. 

        Args:
            x (ndarray): input to model (will be either one image or mini-batch). DIM = (flattened_input_shape,) 
            expected (ndarray): value expected in output (will be either one value or mini-batch). DIM = (flattened_input_shape,)

        Returns:
            float: loss value of the iteration
        """
        out = x
        for l in range(len(self.layers)):
            out = self.layers[l].forward(out)
        loss = self.loss.forward(out, expected)
        return loss

    def backward(self):
        """Backward propagation through the layers.
        """
        delta = self.loss.backward()

        for l in range(1, len(self.layers)):
            delta = self.layers[-l].backward(delta)
    
    def choice(self, proba_vector):
        return np.argmax(proba_vector)
    
    def prediction(self, x, expected):
        out = x
        for l in range(len(self.layers)):
            out = self.layers[l].forward(out)
        return 1 if self.choice(self.softmax(out)) == expected else 0


