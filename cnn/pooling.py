import numpy as np
from cnn.layer import Layer

class MaxPool(Layer):
    """Max Pooling layer.

        Args:
            size_kernel (int): size of the kernels
            stride (int, optional): stride of the convolution (of how much the kernel moves)
        """
    def __init__(self, size_kernel:int, stride:int=None):
        self.size_kernel = size_kernel
        self.out_dim = None
        self.input = None
        self.maximum_indices = None # keep track of maximum indices

        self.stride = stride if stride is not None else size_kernel

    def initial_param(self, dim_in, *args): 
        """Initialize the parameters.

        Args:
            dim_in (tuple): dimensions entering the layer.
        """

        h_in, w_in, channels_in = dim_in
        
        # TODO I might need to check out dim to be good it has to be exact
        out_dim = int(np.floor((h_in - self.size_kernel) / self.stride) + 1)
        
        self.out_dim = (out_dim, out_dim, channels_in)

    def forward(self, x:np.ndarray):
        """Forward pass through max pooling layer.

        Args:
            x (np.ndarray): input DIM = (batch_size, input_height, input_width, number_channels)

        Returns:
            np.ndarray: max pooled
        """
        self.input = x

        w = np.lib.stride_tricks.sliding_window_view(x, (self.size_kernel, self.size_kernel), axis=(1,2)) #window view

        w = w[:, ::self.stride, ::self.stride, :] #apply stride 

        # i need to track max indices for backprop
        self.maximum_indices = np.argmax(w.reshape(x.shape[0], self.out_dim[0], self.out_dim[1], self.out_dim[2], -1), axis=4)   

        return np.max(w, axis=(4, 5))

    def backward(self, dL_dout:np.ndarray, *args): 
        """Back propagation MaxPool layer.

        Args:
            dL_dout (ndarray): gradient from next layer

        Returns:
            ndarray: gradient for layer before
        """
        out = np.zeros_like(self.input)  # output gradient

        # Create indexes for the array
        b_index, h_index, w_index, c_index = np.indices(self.maximum_indices.shape)

        # Get real indexes with stride
        row_add = self.maximum_indices // self.size_kernel
        col_add = self.maximum_indices % self.size_kernel

        real_h = h_index * self.stride + row_add
        real_w = w_index * self.stride + col_add

        # Place gradients at max positions
        out[b_index, real_h, real_w, c_index] = dL_dout

        return out
    
    def create_max_mask(self):
        """I don't need it but helped me understand how to do it. 
        Creates a mask with ones at maximum positions.

        Returns:
            ndarray: mask of maxes
        """
        mask = np.zeros_like(self.input)

        row_add = self.maximum_indices // self.size_kernel
        col_add = self.maximum_indices % self.size_kernel

        b_index, h_index, w_index, c_index = np.indices(self.maximum_indices.shape)

        transformed_h = h_index * self.stride + row_add
        transformed_w = w_index * self.stride + col_add

        mask[b_index, transformed_h, transformed_w, c_index] = 1

        return mask
