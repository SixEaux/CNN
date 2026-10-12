import numpy as np

class Layer:

    def initial_param(self, input_shape: tuple, output_shape: tuple) -> None:
        """Initialize weights/biases for this layer."""
        raise NotImplementedError("Must be implemented by subclass.")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: takes input, returns output."""
        raise NotImplementedError("Must be implemented by subclass.")
    
    def backward(self, dL_dout: np.ndarray, batch_size: int) -> np.ndarray:
        """Backward pass: takes gradient, returns gradient for previous layer."""
        raise NotImplementedError("Must be implemented by subclass.")