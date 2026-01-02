import numpy as np

def he_initialization(dim_in, size): # for relu
    rng = np.random.default_rng()
    return rng.normal(0, np.sqrt(2/dim_in), size)

def xavier_initialization(dim_in, size):
    rng = np.random.default_rng()
    return rng.normal(0, np.sqrt(2/dim_in), size)