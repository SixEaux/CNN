import numpy as np
from cnn.parmeter_initialization import he_initialization, xavier_initialization

class Convolutional:
    """Convolutional layer.

        Args:
            number_kernels (int): number of kernels
            size_kernel (int): size of the kernels

            padding (int): padded to input (how much we add to input's border) (maybe change it to (pad_start, pad_end) in the future). Defaults to 0.
            stride (int, optional): stride of the convolution (of how much the kernel moves). Defaults to 1.
            initialization (str): type of initialization between: he, xavier
        """
    def __init__(self, number_kernels:int, size_kernel:int, stride:int=1, padding:int=0, initialization:str="xavier"):

        self.input = None

        self.number_kernels = number_kernels
        self.size_kernel = size_kernel
        self.out_dim = None # (height_out_conv, width_out_conv, number_kernels)
        self.kernel = None # (size_kernel, size_kernel, channels_in, number_kernels)
        self.bias = None # (1, height_out_conv, width_out_conv, number_kernels)

        self.stride = stride
        self.padding = padding
        self.initialization = initialization

    def initial_param(self, dim_in:tuple): 
        """Initialize the parameters.

        Args:
            dim_in (tuple): dimensions entering the layer.
        """
        h_in, w_in, channels_in = dim_in

        if self.initialization == "he":
            self.kernel = he_initialization(h_in*w_in, self.size_kernel*self.size_kernel*channels_in*self.number_kernels).reshape((self.size_kernel, self.size_kernel, channels_in, self.number_kernels)).astype(np.float32)
        elif self.initialization == "xavier":
            self.kernel = xavier_initialization(h_in*w_in, self.size_kernel*self.size_kernel*channels_in*self.number_kernels).reshape((self.size_kernel, self.size_kernel, channels_in, self.number_kernels)).astype(np.float32)
        elif self.initialization == "random":
            self.weight = np.random.uniform(-1, 1, (self.size_kernel, self.size_kernel, channels_in, self.number_kernels)).astype(np.float32)
        else:
            raise ValueError("I don't know this initialization.")

        #for now supposing same height and width in out and same padding all around
        out_dim = int(np.floor((h_in - self.size_kernel + 2*self.padding) / self.stride) + 1)
        self.out_dim = (out_dim, out_dim, self.number_kernels)
        self.bias = np.zeros((1, out_dim, out_dim, self.number_kernels)).astype(np.float32)

    def convolution_forward_tensordot(self, x:np.ndarray):
        """Convolution forward pass.

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
        return self.convolution_forward_tensordot(x) + self.bias       

    def backward_filter_tensordot(self, dL_dout:np.ndarray):
        """Get the gradient of the error wrt the filter to adjust weights.
        For this gradient I need to compute the convolution(input, error_gradient).
        If I understood correctly you don't take here into account stride and pad,
        this is only taken into account for the computation of the error wrt the input.

        Args:
            dL_dout (np.ndarray): gradient from next layer

        Returns:
            np.ndarray: _description_
        """
        k_h, k_w, k_c, number_k = self.kernel.shape

        if self.padding > 0:
            x = np.pad(self.input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))) # pad the image
        else:
            x = self.input

        w = np.lib.stride_tricks.sliding_window_view(x, (k_h, k_w), axis=(1,2)) #window view

        w = w[:, ::self.stride, ::self.stride, :, :, :] #apply stride

        c = np.tensordot(w, dL_dout, axes=[(0, 1, 2), (0, 1, 2)]) #reduction

        return c.transpose(1,2,0,3)
    
    def backward_input_tensordot(self, dL_dout:np.ndarray):
        """Get the gradient of the error wrt the input.
        For this gradient I need to compute the convolution(180_flip(kernel), error_gradient).
        If I understood correctly here I need to dilate (add rows and columns of zeros in-between) the error_gradient of stride-1 rows and columns.
        And I also need to pad the error of kernel_size-1-pad (to get good full convolution) and then do a valid convolution.

        Args:
            dL_dout (np.ndarray): gradient from next layer

        Returns:
            np.ndarray: gradient for layer before
        """
        #recover correct sizes from stride and pad
        b, h, w, c = dL_dout.shape

        pad_value = self.size_kernel - 1 - self.padding #padding to add

        new_h, new_w = h + (h-1)*(self.stride-1) + 2*pad_value, w + (w-1)*(self.stride-1) + 2*pad_value #h, w conversion to take stride (add zeros between rows) and padding (pad) into account

        out = np.zeros((b, new_h, new_w, c))

        out[:, pad_value:new_h-pad_value:self.stride, pad_value:new_w-pad_value:self.stride, :] = dL_dout

        #Valid convolution with filter rotated

        k_h, k_w, k_c, number_k = self.kernel.shape

        rot_kernel = np.rot90(self.kernel, k=2, axes=(0,1))

        w = np.lib.stride_tricks.sliding_window_view(out, (k_h, k_w), axis=(1,2)) #window view (! window of size kernel because we want to reduce by w_out, h_out and c_out in order to keep kernel shapes)

        c = np.tensordot(w, rot_kernel, axes=[(3, 4, 5), (3, 0, 1)]) #reduction

        return c

    def backward(self, dL_dout:np.ndarray, learning_rate:float, batch_size:int=1):
        """Chef backpropagation convolutional layer. It also adjustes the filters and biases

        Args:
            dL_dout (np.ndarray): gradient from next layer
            batch_size (int, optional): size of the batch. Defaults to 1.
            learning_rate (float): learning rate

        Returns:
            np.ndarray: gradient error wrt input for layer before
        """        
        dL_df = self.backward_filter_tensordot(dL_dout)
        dL_db = np.sum(dL_dout, axis=0, keepdims=True) #sum accross batches

        dL_dx = self.backward_input_tensordot(dL_dout)
        
        self.kernel -= learning_rate * dL_df / batch_size
        self.bias -= learning_rate * dL_db / batch_size

        return dL_dx
    