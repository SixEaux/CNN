import numpy as np

class Layer:

    def initial_param(self, input_shape: tuple, output_shape: tuple) -> None:
        """Initialize weights/biases for this layer."""
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: takes input, returns output."""
        pass
    
    def backward(self, dL_dout: np.ndarray, batch_size: int) -> np.ndarray:
        """Backward pass: takes gradient, returns gradient for previous layer."""
        pass