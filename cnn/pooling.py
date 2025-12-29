import numpy as np

class MaxPool:
    """Max Pooling layer.

        Args:
            size_kernel (int): size of the kernels
            stride (int, optional): stride of the convolution (of how much the kernel moves). Defaults to 1.
        """
    def __init__(self, size_kernel:int, stride:int=None):
        self.size_kernel = size_kernel
        self.out_dim = None
        self.input = None
        self.maximum_indices = None
        self.max_mask = None

        self.stride = stride if stride is not None else size_kernel

    def initial_param(self, dim_in): 
        """Initialize the parameters.

        Args:
            dim_in (tuple): dimensions entering the layer.
        """

        #for now supposing same height and width in out and same padding all around
        h_in, w_in, channels_in = dim_in
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
    
    def create_max_mask(self):
        mask = np.zeros_like(self.input)

        row_add = self.maximum_indices // self.size_kernel
        col_add = self.maximum_indices % self.size_kernel

        b_index, h_index, w_index, c_index = np.indices(self.maximum_indices.shape)

        transformed_h = h_index * self.stride + row_add
        transformed_w = w_index * self.stride + col_add

        mask[b_index, transformed_h, transformed_w, c_index] = 1

        return mask

    def backward(self, dL_dout, batch_size=1): 
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

class MeanPool:
    """Mean Pooling layer.

        Args:
            size_kernel (int): size of the kernels
            padding (int): padded to input (how much we add to input's border) (maybe change it to (pad_start, pad_end) in the future). Defaults to 0.
            stride (int, optional): stride of the convolution (of how much the kernel moves). Defaults to 1.
        """
    def __init__(self, size_kernel:int):
        self.size_kernel = size_kernel
        self.out_dim = None
        self.input = None

        self.stride = size_kernel #stride = size_kernel it is a pain to do it other way

    def initial_param(self, dim_in): 
        """Initialize the parameters.

        Args:
            dim_in (tuple): dimensions entering the layer.
        """

        #for now supposing same height and width in out and same padding all around
        h_in, w_in, channels_in = dim_in
        out_dim = int(np.floor((h_in - self.size_kernel) / self.stride) + 1)
        
        self.out_dim = (out_dim, out_dim, channels_in)

    def forward(self, x:np.ndarray):
        """Forward pass through mean pooling layer.

        Args:
            x (np.ndarray): input DIM = (batch_size, input_height, input_width, number_channels)

        Returns:
            np.ndarray: mean pooled
        """
        self.input = x

        w = np.lib.stride_tricks.sliding_window_view(x, (self.size_kernel, self.size_kernel), axis=(1,2)) #window view
        # i need to track max indices for backprop

        w = w[:, ::self.stride, ::self.stride, :] #apply stride

        return np.mean(w, axis=(4, 5))
    
    def backward(self, dL_dout:np.ndarray):
        """Backpropagation / unpooling with mean.

        Args:
            dL_dout (np.ndarray): delta from next layer. DIM = (batch_size, out_dim, out_dim_ channels_in)
        """
        dL_dout = dL_dout / self.size_kernel**2
        out = np.repeat(dL_dout, self.size_kernel, axis=1) # repeat the kernel by axis 1
        return np.repeat(out, self.size_kernel, axis=2) # repeat the kernel by axis 2
