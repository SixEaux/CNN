import numpy as np

class MaxPool:
    """Max Pooling layer.

        Args:
            size_kernel (int): size of the kernels
            padding (int): padded to input (how much we add to input's border) (maybe change it to (pad_start, pad_end) in the future). Defaults to 0.
            stride (int, optional): stride of the convolution (of how much the kernel moves). Defaults to 1.
        """
    def __init__(self, size_kernel:int, stride:int, padding:int=0):
        self.size_kernel = size_kernel
        self.out_dim = None
        self.input = None
        self.max_indices = None

        self.stride = stride
        self.padding = padding

    def initial_param(self, dim_in): 
        """Initialize the parameters.

        Args:
            dim_in (tuple): dimensions entering the layer.
        """

        #for now supposing same height and width in out and same padding all around
        h_in, w_in, channels_in = dim_in
        out_dim = int(((h_in - self.size_kernel + 2*self.padding) / self.stride) + 1)
        
        self.out_dim = (out_dim, out_dim, channels_in)

    def forward(self, x:np.ndarray):
        """Forward pass through max pooling layer.

        Args:
            x (np.ndarray): input DIM = (batch_size, input_height, input_width, number_channels)

        Returns:
            np.ndarray: max pooled
        """
        self.input = x

        if self.padding > 0:
            p = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))) # pad the image
        else:
            p = x

        w = np.lib.stride_tricks.sliding_window_view(p, (self.size_kernel, self.size_kernel), axis=(1,2)) #window view
        # i need to track max indices for backprop

        w = w[:, ::self.stride, ::self.stride, :] #apply stride      

        return np.max(w, axis=(4, 5))
    
    def backward(self, dL_dout):
        pass




class MeanPool:
    """Mean Pooling layer.

        Args:
            size_kernel (int): size of the kernels
            padding (int): padded to input (how much we add to input's border) (maybe change it to (pad_start, pad_end) in the future). Defaults to 0.
            stride (int, optional): stride of the convolution (of how much the kernel moves). Defaults to 1.
        """
    def __init__(self, size_kernel:int, stride:int, padding:int=0):
        self.size_kernel = size_kernel
        self.out_dim = None
        self.input = None

        self.stride = stride
        self.padding = padding

    def initial_param(self, dim_in): 
        """Initialize the parameters.

        Args:
            dim_in (tuple): dimensions entering the layer.
        """

        #for now supposing same height and width in out and same padding all around
        h_in, w_in, channels_in = dim_in
        out_dim = int(((h_in - self.size_kernel + 2*self.padding) / self.stride) + 1)
        
        self.out_dim = (out_dim, out_dim, channels_in)

    def forward(self, x:np.ndarray):
        """Forward pass through mean pooling layer.

        Args:
            x (np.ndarray): input DIM = (batch_size, input_height, input_width, number_channels)

        Returns:
            np.ndarray: mean pooled
        """
        self.input = x

        if self.padding > 0:
            p = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))) # pad the image
        else:
            p = x

        w = np.lib.stride_tricks.sliding_window_view(p, (self.size_kernel, self.size_kernel), axis=(1,2)) #window view
        # i need to track max indices for backprop

        w = w[:, ::self.stride, ::self.stride, :] #apply stride

        return np.mean(w, axis=(4, 5))
    
    def backward(self, dL_dout):
        pass