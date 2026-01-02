import numpy as np



# IM2COL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def convolution_forward_im2col(self, x:np.ndarray):
        """Here is anot5her method of convolution which should be faster.
        It consists of creating the image with patches and reshape it and the kernel to vectorized form
        then np.dot should get the convolution.

        I do not like a lot using as_strided but it is the fastest.

        Args:
            x (np.ndarray): input array DIM = (batch_size, h, w, c)
            kernel (np.ndarray): kernel DIM = (size_kernel, size_kernel, c_in, number_kernels)
            padding (int): pad image 
            stride (int): stride image
        
        Returns:
            np.ndarray: convolution output
        """

        if self.padding > 0:
            x_pad = np.pad(x, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)))
        else:
            x_pad = x        

        b, h, w, c = x_pad.shape
        k_h, k_w, k_c, k_n = self.kernel.shape
        o_h, o_w = ((h-k_h)//self.stride)+1, ((w-k_w)//self.stride)+1

        x_patches = np.lib.stride_tricks.as_strided(x_pad, (b, o_h, o_w, k_h, k_w, k_c), 
                                                    (x_pad.strides[0], x_pad.strides[1]*self.stride, x_pad.strides[2]*self.stride, x_pad.strides[1], x_pad.strides[2], x_pad.strides[3]))

        out = x_patches.reshape((b*o_h*o_w, k_h*k_w*k_c))
        kernel_reshaped = self.kernel.reshape((k_h*k_w*k_c, k_n))

        return (out @ kernel_reshaped).reshape(b, o_h, o_w, k_n)


def backward_filter_im2col(self, dL_dout:np.ndarray):
    k_h, k_w, k_c, number_k = self.kernel.shape

    if self.padding > 0:
        x = np.pad(self.input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))) # pad the image
    else:
        x = self.input

    b, h, w, c = x.shape
    k_h, k_w, k_c, k_n = self.kernel.shape
    o_h, o_w = ((h-k_h)//self.stride)+1, ((w-k_w)//self.stride)+1

    w = np.lib.stride_tricks.as_strided(x, (b, o_h, o_w, k_h, k_w, k_c), 
                                                    (x.strides[0], x.strides[1]*self.stride, x.strides[2]*self.stride, x.strides[1], x.strides[2], x.strides[3]))
    
    dL_dout = dL_dout.reshape((b*o_h*o_w, k_n))

    w_strided = w.reshape((b*o_h*o_w, k_h*k_w*k_c))

    return (w_strided.T @ dL_dout).reshape((k_h, k_w, k_c, k_n))



def backward_input_im2col(self, dL_dout:np.ndarray): #this makes the interpreter explode and stops so use with tensordot instead
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
        k_h, k_w, k_c, k_n = self.kernel.shape

        pad_value = k_h - 1 - self.padding #padding to add

        new_h, new_w = h + (h-1)*(self.stride-1) + 2*pad_value, w + (w-1)*(self.stride-1) + 2*pad_value #h, w conversion to take stride (add zeros between rows) and padding (pad) into account

        out = np.zeros((b, new_h, new_w, c))

        out[:, pad_value:new_h-pad_value:self.stride, pad_value:new_w-pad_value:self.stride, :] = dL_dout

        #Valid convolution with filter rotated

        b, h, w, c = out.shape
        i_b, i_h, i_w, i_c = self.input.shape

        rot_kernel = np.rot90(self.kernel, k=2, axes=(0,1)).transpose((0, 1, 3, 2)).reshape((-1, k_c))

        o_h, o_w = (new_h-k_h)+1, (new_w-k_w)+1

        out_patch = np.lib.stride_tricks.as_strided(out, (b, o_h, o_w, k_h, k_w, k_n), 
                                                    (out.strides[0], out.strides[1]*self.stride, out.strides[2]*self.stride, out.strides[1], out.strides[2], out.strides[3]))

        out_patch_reshaped = out_patch.reshape((b*o_h*o_w, k_h*k_w*k_n))

        return (out_patch_reshaped @ rot_kernel).reshape((i_b, i_h, i_w, i_c))

# IM2COL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
