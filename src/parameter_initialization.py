import numpy as np

def he_initialization(dim_in, size): # for relu
    """He initialization. Initialize by doing a normal with 
    mu=0 and sd=sqrt(2/input_dimension).

    Args:
        dim_in (int): input dimension (for dense flattened input shape and for dense h_in*w_in)
        size (_type_): size of the normal to generate so either of the weights or the kernel

    Returns:
        ndarray: initialization of weights and kernel
    """
    rng = np.random.default_rng()
    return rng.normal(0, np.sqrt(2/dim_in), size)

def xavier_initialization(dim_in, dim_out, size): # for sigmoid
    """Xavier initialization. Initialize by doing a normal with 
    mu=0 and sd=sqrt(2/input_dimension).

    Args:
        dim_in (int): input dimension (for dense flattened input shape and for dense h_in*w_in)
        size (_type_): size of the normal to generate so either of the weights or the kernel

    Returns:
        ndarray: initialization of weights and kernel
    """
    rng = np.random.default_rng()
    value = np.sqrt(6/(dim_in + dim_out))
    return rng.uniform(-value, value, size)