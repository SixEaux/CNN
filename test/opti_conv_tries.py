import numpy as np


def convolution_sliding_tensordot(x:np.ndarray, kernel:np.ndarray, padding:int=0, stride:int=1):
        """Convolution forward pass.

        For output height and width (I) : O = ((I - K + P_start + P_end) / S) + 1 

        Args:
            x (np.ndarray): input DIM = (batch_size, input_height, input_width, number_channels)
    
        Returns:
            np.ndarray: output after convolution DIM = (batch_size, output_height, output_width, number_kernels)
        """
        k_h, k_w, k_c, number_k = kernel.shape

        if padding > 0:
            p = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0))) # pad the image
        else:
            p = x

        assert k_c == p.shape[3], "Channel dimension mismatch" 

        w = np.lib.stride_tricks.sliding_window_view(p, (k_h, k_w), axis=(1,2)) #window view

        w = w[:, ::stride, ::stride, :] #apply stride

        c = np.tensordot(w, kernel, axes=[(3, 4, 5), (2, 0, 1)]) #reduction

        return c


def convolution_im2col_samekernelstride(x:np.ndarray, kernel:np.ndarray, padding:int=0, stride:int=None):
        """Here is anot5her method of convolution which should be faster.
        It consists of creating the image with patches and reshape it and the kernel to vectorized form
        then np.dot should get the convolution.
        I will suppose for now that stride == kernel_size to get the main idea.

        when this works well i actually can just keep everything in vectorized shapes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        see here for great explanation: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

        Args:
            x (np.ndarray): input array DIM = (batch_size, h, w, c)
            kernel (np.ndarray): kernel DIM = (size_kernel, size_kernel, c_in, number_kernels)
        
        Returns:
            np.ndarray: convolution output
        """

        if padding > 0:
            x_pad = np.pad(x, ((0,0), (padding, padding), (padding, padding), (0,0)))
        else:
            x_pad = x        

        b, h, w, c = x_pad.shape
        k_h, k_w, k_c, k_n = kernel.shape
        o_b, o_h, o_w, o_c = b, ((h-k_h)//k_h)+1, ((w-k_w)//k_w)+1, k_n

        x_reshaped_1 = x_pad.reshape((b, h//k_h, k_h, w//k_w, k_w, c)).transpose((0, 1, 3, 2, 4, 5)) #create array of patches

        x_reshaped_2 = x_reshaped_1.reshape((b, h//k_h, w//k_w, -1)) #vectorize the input shapes

        kernel_reshaped = kernel.reshape((k_h*k_w*k_c, k_n))

        out = np.dot(x_reshaped_2, kernel_reshaped) #dot product by last dimension

        out_reshaped = out.reshape((o_b, o_h, o_w, o_c))    

        return out_reshaped

def convolution_im2col_notpossible(x:np.ndarray, kernel:np.ndarray, padding:int=0, stride:int = 1): #doesnt work because impossible to do x_pad[b_out, h_start:h_end, w_start:w_end, :] with arrays
     
    if padding > 0:
            x_pad = np.pad(x, ((0,0), (padding, padding), (padding, padding), (0,0)))
    else:
        x_pad = x        

    b, h, w, c = x_pad.shape
    k_h, k_w, k_c, k_n = kernel.shape
    o_b, o_h, o_w, o_c = b, ((h-k_h)//stride)+1, ((w-k_w)//stride)+1, k_n

    # get indices output
    b_out, h_out, w_out, c_out = np.indices((o_b, o_h, o_w, o_c))
    out = np.zeros((o_b, o_h, o_w, o_c))

    #get indices of kernel view
    h_start = h_out * stride
    w_start = w_out * stride

    h_end = h_out * stride + k_h
    w_end = w_out * stride + k_w

    # reshape the kernel
    kernel_reshaped = kernel.reshape((k_h*k_w*k_c, k_n))

    #calculate output
    out[b_out, h_out, w_out, c_out] = (x_pad[b_out, h_start:h_end, w_start:w_end, :].reshape((o_b, -1)) @ kernel_reshaped[:, c_out])


def convolution_im2col_loop(x:np.ndarray, kernel:np.ndarray, padding:int=0, stride:int = 1):
    if padding > 0:
        b, h, w, c = x.shape
        x_pad = np.zeros((b, h+2*padding, w+2*padding, c))
        b, h, w, c = x_pad.shape
        x_pad[:, padding:h-padding, padding:w-padding, :] = x
    else:
        x_pad = x        
        b, h, w, c = x_pad.shape

    
    k_h, k_w, k_c, k_n = kernel.shape
    o_b, o_h, o_w, o_c = b, ((h-k_h)//stride)+1, ((w-k_w)//stride)+1, k_n

    out = np.zeros((o_b, o_h, o_w, o_c))
    kernel_reshaped = kernel.reshape((k_h*k_w*k_c, k_n))


    for h_l in range(o_h):
        for w_l in range(o_w):
            #get indices of kernel
            h_start = h_l * stride
            w_start = w_l * stride

            h_end = h_l * stride + k_h
            w_end = w_l * stride + k_w

            out[:, h_l, w_l, :] = x_pad[:, h_start:h_end, w_start:w_end, :].reshape((o_b, -1)) @ kernel_reshaped
    
    return out





def convolution_im2col_as_strided(x:np.ndarray, kernel:np.ndarray, padding:int=0, stride:int=1):
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

        if padding > 0:
            x_pad = np.pad(x, ((0,0), (padding, padding), (padding, padding), (0,0)))
        else:
            x_pad = x        

        b, h, w, c = x_pad.shape
        k_h, k_w, k_c, k_n = kernel.shape
        o_h, o_w = ((h-k_h)//stride)+1, ((w-k_w)//stride)+1

        x_patches = np.lib.stride_tricks.as_strided(x_pad, (b, o_h, o_w, k_h, k_w, k_c), 
                                                    (x_pad.strides[0], x_pad.strides[1]*stride, x_pad.strides[2]*stride, x_pad.strides[1], x_pad.strides[2], x_pad.strides[3]))

        out = x_patches.reshape((b*o_h*o_w, k_h*k_w*k_c))
        kernel_reshaped = kernel.reshape((k_h*k_w*k_c, k_n))

        return (out @ kernel_reshaped).reshape(b, o_h, o_w, k_n)



def convolution_im2col_indexing(x:np.ndarray, kernel:np.ndarray, padding:int=0, stride:int = 1):
     
    if padding > 0:
            x_pad = np.pad(x, ((0,0), (padding, padding), (padding, padding), (0,0)))
    else:
        x_pad = x        

    b, h, w, c = x_pad.shape
    k_h, k_w, k_c, k_n = kernel.shape
    o_b, o_h, o_w, o_c = b, ((h-k_h)//stride)+1, ((w-k_w)//stride)+1, k_n

    # output positions
    out_h = np.arange(o_h) * stride
    out_w = np.arange(o_w) * stride

    #kernel positions
    kernel_h = np.arange(k_h)
    kernel_w = np.arange(k_w)

    real_rows = out_h[:, None] + kernel_h[None, :] # (o_h, k_h)
    real_cols = out_w[:, None] + kernel_w[None, :] # (o_w, k_w)

    x_patch = x_pad[:, real_rows[:, None, :, None], real_cols[None, :, None, :], :] #(b, o_h, o_w, k_h, k_w, k_c)

    x_patch = x_patch.reshape((b*o_h*o_w, k_h*k_w*k_c))

    kernel_reshaped = kernel.reshape((k_h*k_w*k_c, k_n))

    return (x_patch @ kernel_reshaped).reshape((b, o_h, o_w, k_n))



# x = np.arange(16).reshape(1, 4, 4, 1)
# k = np.arange(9).reshape((3,3, 1, 1))


# assert np.array_equal(convolution_im2col_as_strided(x, k, 2, 1), convolution_im2col_loop(x, k, 2, 1))



"""x = np.random.randn(8, 64, 64, 3).astype(np.float32)
k = np.random.randn(3, 3, 3, 32).astype(np.float32)

import time
t0 = time.time()
for i in range(1000):
    _ = convolution_im2col_as_strided(x, k, padding=3, stride=1)
t1 = time.time()

print("Time:", t1 - t0)"""