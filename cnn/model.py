"""IMPORTANT INFO : 
    - input data is (batch_size, height, width, number_channels) (if no channels number_cahnnels = 1)
    - then when flattened becomes (batch_size, height * width * number_channels) (if without batches => batch_size = 1)
    - following this the data when imported will be (number_images, height, width, number_channels)
"""
"""
To facilitate passing parameters -> to create a model you need to:
    Model(
    layers = [Convolutional(), Activation(), Flatening(), Pooling(), Dense()], 
    other_parameters)
"""

"""
As a convention when i use:
- x it is an input
- C or L cost / loss
- w weights
- b biais
- z = wx + b
- a = activ(z)
"""

import numpy as np
import pickle
import os

from cnn.loss import Loss
from cnn.dense import Dense
from cnn.convolution import Convolutional
from cnn.flattening import Flattening
from cnn.pooling import MaxPool, MeanPool



class Model:
    """Model NN.

        Args:
            layers (list[Layers]): list of layers (objects)
            loss (Loss): loss object 
            dataset (str): dataset used
            batch_size (int): size of batch used
        """

    def __init__(self, layers: list, loss: Loss, dataset: str):
        self.layers = layers
        self.loss = loss
        self.dataset = dataset
        
        self.model_initial()

    def model_initial(self): 
        """Initialize the model based on dimensions input.
        """
        dims_dataset = {"mnist": (28, 28, 1)}
        last_dim_out = dims_dataset[self.dataset]

        for l in self.layers:
            l.initial_param(last_dim_out)

            if isinstance(l, Dense):
                last_dim_out = l.number_neurons
            elif isinstance(l, Convolutional):
                last_dim_out = l.out_dim
            elif isinstance(l, MaxPool) or isinstance(l, MeanPool):
                last_dim_out = l.out_dim
            elif isinstance(l, Flattening):
                last_dim_out = last_dim_out[0]*last_dim_out[1]*last_dim_out[2]

    def forward(self, x:np.ndarray, expected:np.ndarray):
        """Forward propagation through all the layers. 

        Args:
            x (ndarray): input to model (will be either one image or mini-batch). DIM = (batch_size, flattened_input_shape) 
            expected (ndarray): value expected in output (will be either one value or mini-batch). DIM = (number_classes,)

        Returns:
            float: loss value of the iteration
        """

        out = x
        for l in self.layers:
            out = l.forward(out)
        loss = self.loss.forward(out, expected)
        return loss

    def backward(self, batch_size:int, learning_rate:float):
        """Backward propagation through the layers.
        """
        delta = self.loss.backward()

        for l in reversed(self.layers):
            delta = l.backward(delta, learning_rate, batch_size)

    def choice(self, probabilities:np.ndarray):
        """Choose from outputs the one with higher "probability" (logits or smthg like this).

        Args:
            probabilities (ndarray): vector of "probabilities". DIM = (batch_size, number_classes)

        Returns:
            ndarray: position of highest probability. DIM = (batch_size, 1)
        """

        return np.argmax(probabilities, axis=1, keepdims=True)

    def prediction(self, x:np.ndarray):
        """Prediction for an image.

        Args:
            x (ndarray): input image. DIM = input_shape 

        Returns:
            ndarray: prediction. DIM = (batch_size, 1)
        """
        out = x
        for l in range(len(self.layers)):
            out = self.layers[l].forward(out)
        return self.choice(out)
    
    def export_model(self, filename:str):
        safe = {"layers":self.layers, "loss":self.loss, "dataset":self.dataset}
        with open(os.path.join("trained_models" + filename), "wb") as f:
            pickle.dump(safe, f)
