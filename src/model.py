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

from src.loss import Loss
from src.dense import Dense
from src.convolution import Convolutional
from src.flattening import Flattening
from src.pooling import MaxPool
from src.dropout import Dropout
from src.import_data import import_data

from src.layer import Layer


class Model:
    """Model NN.

    Args:
        layers (list[Layers]): list of layers (objects)
        loss (Loss): loss object
        dataset (str): dataset used
        batch_size (int): size of batch used
    """

    def __init__(
        self,
        layers: list[Layer],
        loss: Loss,
        dataset: str,
        initialized: bool = False,
        CAM_image: np.ndarray = None,
    ):
        self.layers = layers
        self.loss = loss
        self.dataset = dataset
        _, _, _, _, _, _, self.labels = import_data(self.dataset)

        self.saved_outputs = []  # to save some outputs from layers

        self.input_size = None

        if CAM_image is not None:
            self.CAM_image = CAM_image
            self.saved_gradients = [
                [] for _ in self.CAM_image
            ]  # to save the gradients of some images during training
        else:
            self.CAM_image = CAM_image

        if not initialized:
            self.model_initial()

    def precompute_fan_out(self, l, dim_in: tuple):
        if isinstance(l, Dense):
            return l.number_neurons
        elif isinstance(l, Convolutional):
            h_in, w_in, channels_in = dim_in
            h_out = int(np.floor((h_in - l.size_kernel + 2 * l.padding) / l.stride) + 1)
            return h_out, h_out, l.number_kernels
        else:
            return

    def model_initial(self):
        """Initialize the model based on dimensions input."""
        dims_dataset = {"mnist": (28, 28, 1), "fashion_mnist": (28, 28, 1)}
        last_dim_out = dims_dataset[self.dataset]
        self.input_size = last_dim_out

        for l in self.layers:

            fan_out = self.precompute_fan_out(l, last_dim_out)

            l.initial_param(last_dim_out, fan_out)

            if isinstance(l, Dense):
                last_dim_out = l.number_neurons
            elif isinstance(l, Convolutional):
                last_dim_out = l.out_dim
            elif isinstance(l, MaxPool):
                last_dim_out = l.out_dim
            elif isinstance(l, Flattening):
                last_dim_out = last_dim_out[0] * last_dim_out[1] * last_dim_out[2]

    def forward(
        self,
        x: np.ndarray,
        expected: np.ndarray = None,
        test: bool = False,
        save_out: bool = False,
    ):
        """Forward propagation through all the layers.

        Args:
            x (ndarray): input to model (will be either one image or mini-batch). DIM = (batch_size, flattened_input_shape)
            expected (ndarray): value expected in output (will be either one value or mini-batch). DIM = (number_classes, 1)

        Returns:
            tuple: output DIM = (batch_size, number_classes) and loss value of the iteration DIM = (batch_size, 1)
        """

        assert x.ndim == 4, "Not the right shapes for the input"
        self.saved_outputs = []

        out = x
        for l in self.layers:

            if test and isinstance(
                l, Dropout
            ):  # if its a test then dropout not taking into account
                continue

            out = l.forward(out)

            if save_out:
                self.saved_outputs.append(out)

        if expected is not None:
            loss = self.loss.forward(out, expected)
            return out, loss
        else:
            return out, None

    def backward(
        self,
        batch_size: int,
        learning_rate: float,
        momentum_rate: float,
        save: list = None,
    ):
        """Backward propagation through the layers.

        Args:
            batch_size (int): size of the batch
            learning_rate (float): learning rate
            momentum_rate (float): dependence on gradient before
        """
        delta = self.loss.backward()

        for l in reversed(self.layers):
            delta = l.backward(delta, learning_rate, momentum_rate, batch_size)

        if save is not None:
            for i in range(len(save)):
                if save[i] is not None:
                    self.saved_gradients[i].append(delta[save[i]])

    def choice(self, probabilities: np.ndarray):
        """Choose from outputs the one with higher "probability" (logits or smthg like this).

        Args:
            probabilities (ndarray): vector of "probabilities". DIM = (batch_size, number_classes)

        Returns:
            ndarray: position of highest probability. DIM = (batch_size, 1)
        """

        return np.argmax(probabilities, axis=1, keepdims=True)
