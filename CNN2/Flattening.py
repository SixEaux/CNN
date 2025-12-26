class Flattening:
    def __init__(self):
        self.x = None
    
    def initial_param(self, dim_in): # just so every layer has one
        return

    def forward(self, x):
        """Reshape input for fully connected layer.

        Args:
            x (ndarray): input (batch_size, h, w, c)

        Returns:
            ndarray: reshaped to (batch_size, h*w*c)
        """
        self.x = x
        return x.reshape((x.shape[0], -1))
    
    def backward(self):
        pass