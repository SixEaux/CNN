"""IMPORTANT INFO : 
    - input data is (batch_size, height, width, number_channels)
    - then when flattened becomes (batch_size, height * width * number_channels) (if without batches => batch_size = 1)
    - following this the data when imported will be (number_images, height, width, number_channels)
"""

import numpy as np
from CNN2.Loss import Loss

class Model:
    """Model NN.

        Args:
            layers (list[Layers]): list of layers (objects)
            loss (Loss): loss object 
            dataset (str): dataset used
        """
    def __init__(self, layers:list, loss:Loss, dataset:str):
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

    def forward(self, x, expected):
        """Forward propagation through all the layers. 

        Args:
            x (ndarray): input to model (will be either one image or mini-batch). DIM = (batch_size, flattened_input_shape) 
            expected (ndarray): value expected in output (will be either one value or mini-batch). DIM = (number_classes,)

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

        for l in reversed(self.layers):
            delta = l.backward(delta)
    
    def choice(self, proba_vector):
        """Choose from outputs the one with higher "probability" (logits or smthg like this).

        Args:
            proba_vector (ndarray): vector of "probabilities". DIM = (number_classes, 1)

        Returns:
            int: position of highest probability
        """
        return np.argmax(proba_vector, axis=0)
    
    def prediction(self, x):
        """Prediction for an image.

        Args:
            x (ndarray): input image. DIM = input_shape

        Returns:
            int: prediction
        """
        out = x
        for l in range(len(self.layers)):
            out = self.layers[l].forward(out)
        return self.choice(out)
    
    def softmax(self, x): # if i want to inspect the probabilities
        """Softmax function.

        Args:
            x (ndarray): input. DIM = (number_neurons_last_layer,)

        Returns:
            ndarray: output. DIM = (number_neurons_last_layer,)
        """
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return  exp_x / np.sum(exp_x, axis=0, keepdims=True)


