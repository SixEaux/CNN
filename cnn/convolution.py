import numpy as np

class Convolutional:
    """Convolutional layer.

        Args:
            number_kernels (int): number of kernels
            stride (int): stride of the convolution (of how much the kernel moves)
            learning_rate (float): learning rate
        """
    def __init__(self, number_kernels:int, stride:int, learning_rate:float):

        self.number_kernels = number_kernels
        self.kernels = None # (height, width, channels_in, number_kernels)
        self.stride = stride

        self.learning_rate = learning_rate

    def convolution(self, x):
        pass
        
