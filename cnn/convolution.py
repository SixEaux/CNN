import numpy as np

class Convolutional:
    """Convolutional layer.

        Args:
            number_kernels (int): number of kernels
            size_kernel (int): size of the kernels

            padding (int): padded to input (how much we add to input's border) (maybe change it to (pad_start, pad_end) in the future). Defaults to 0.
            stride (int, optional): stride of the convolution (of how much the kernel moves). Defaults to 1.

            learning_rate (float): learning rate
        """
    def __init__(self, number_kernels:int, size_kernel:int, learning_rate:float, stride:int=1, padding:int=0):

        self.input = None

        self.number_kernels = number_kernels
        self.size_kernel = size_kernel
        self.out_dim = None
        self.kernel = None # (size_kernel, size_kernel, channels_in, number_kernels)
        self.bias = None # (1, height_out_conv, width_out_conv, number_kernels)

        self.stride = stride
        self.padding = padding

        self.learning_rate = learning_rate

    def initial_param(self, dim_in:tuple): 
        """Initialize the parameters.

        Args:
            dim_in (tuple): dimensions entering the layer.
        """
        h_in, w_in, channels_in = dim_in
        self.kernel = np.random.uniform(-1, 1, (self.size_kernel, self.size_kernel, channels_in, self.number_kernels))

        #for now supposing same height and width in out and same padding all around
        out_dim = int(((h_in - self.size_kernel + 2*self.padding) / self.stride) + 1)
        self.out_dim = (out_dim, out_dim, self.number_kernels)
        self.bias = np.random.uniform(-1, 1, (1, out_dim, out_dim, self.number_kernels))


    def convolution_forward(self, x:np.ndarray):
        """Convolution forward pass (one kernel for the moment).

        For output height and width (I) : O = ((I - K + P_start + P_end) / S) + 1 

        Args:
            x (np.ndarray): input DIM = (batch_size, input_height, input_width, number_channels)
    
        Returns:
            np.ndarray: output after convolution DIM = (batch_size, output_height, output_width, number_kernels)
        """
        k_h, k_w, k_c, number_k = self.kernel.shape

        if self.padding > 0:
            p = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))) # pad the image
        else:
            p = x

        assert k_c == p.shape[3], "Channel dimension mismatch" 

        w = np.lib.stride_tricks.sliding_window_view(p, (k_h, k_w), axis=(1,2)) #window view

        w = w[:, ::self.stride, ::self.stride, :] #apply stride

        c = np.tensordot(w, self.kernel, axes=[(3, 4, 5), (2, 0, 1)]) #reduction

        return c

    def forward(self, x:np.ndarray):
        """Forward pass convolutional layer.

        Args:
            x (np.ndarray): input to layer DIM = (batch_size, input_height, input_width, number_channels)

        Returns:
            np.ndarray: output DIM = (batch_size, output_height, output_width, number_kernels)
        """
        self.input = x
        return self.convolution_forward(x) + self.bias 
    
    def backward(self, dL_dout):
        pass

        
        
