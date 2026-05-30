"""IMPORTANT INFO :
- input data is (batch_size, height, width, number_channels) (if no channels number_cahnnels = 1)
- then when flattened becomes (batch_size, height * width * number_channels) (if without batches => batch_size = 1)
- following this the data when imported will be (number_images, height, width, number_channels)
"""

import numpy as np

from src.loss import Loss
from src.dense import Dense
from src.convolution import Convolutional
from src.flattening import Flattening
from src.pooling import MaxPool
from src.dropout import Dropout
from src.import_data import import_data
from src.cam_image import CAM_IMAGE

from src.layer import Layer
from src.ascii_layers import print_network


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
        cam: CAM_IMAGE = None,
    ):
        self.layers = layers
        self.loss = loss
        self.dataset = dataset
        data = import_data(self.dataset)
        self.labels = data.labels

        self.cam = cam  # CAM_IMAGE instance for tracking/visualization

        self.input_size = None

        self.model_initial()

        print_network(self.layers, self.loss)

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
        record_cam: bool = False,
    ):
        """Forward propagation through all the layers.

        Args:
            x (ndarray): input to model (will be either one image or mini-batch). DIM = (batch_size, flattened_input_shape)
            expected (ndarray): value expected in output (will be either one value or mini-batch). DIM = (number_classes, 1)
            test (bool): if True, skip dropout layers
            record_cam (bool): if True and CAM is initialized, record outputs for CAM

        Returns:
            tuple: output DIM = (batch_size, number_classes) and loss value of the iteration DIM = (batch_size, 1)
        """

        assert x.ndim == 4, "Not the right shapes for the input"

        if record_cam and self.cam is not None:
            self.cam.start_recording()

        out = x
        for l in self.layers:

            if test and isinstance(l, Dropout):  # if its a test then dropout not taken into account
                continue

            out = l.forward(out)

            if record_cam and self.cam is not None:
                self.cam.record_output(out)

        if expected is not None:
            loss = self.loss.forward(out, expected)
            return out, loss
        else:
            return out, None

    def backward(
        self,
        batch_size: int,
        record_cam: bool = False,
        batch_images: np.ndarray = None,
    ):
        """Backward propagation through the layers.

        Args:
            batch_size (int): size of the batch
            record_cam (bool): if True and CAM is initialized, record gradients for CAM
            batch_images (np.ndarray): the batch images. Required if record_cam=True.
                                       Shape: (batch_size, height, width, channels)
        """
        delta = self.loss.backward()

        for l in reversed(self.layers):
            delta = l.backward(delta, batch_size)

        if record_cam and self.cam is not None and batch_images is not None:
            self.cam.record_gradient(batch_images, delta)

    def choice(self, probabilities: np.ndarray):
        """Choose from outputs the one with higher "probability" (logits or smthg like this).

        Args:
            probabilities (ndarray): vector of "probabilities". DIM = (batch_size, number_classes)

        Returns:
            ndarray: position of highest probability. DIM = (batch_size, 1)
        """

        return np.argmax(probabilities, axis=1, keepdims=True)
