from src.layer import Layer
import numpy as np
from src.dense import Dense
from src.convolution import Convolutional

class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, learning_rate: float, decay_rate: float = 0.0, decay_steps: int = 1):
        self.base_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def get_learning_rate(self, epoch: int) -> float:
        """Compute current learning rate with decay."""
        if self.decay_rate > 0:
            if self.lr_decay_method == "exponential":
                return self.base_learning_rate * np.exp(-self.decay_rate * epoch)
            elif self.lr_decay_method == "inverse":
                return self.base_learning_rate / (1 + self.decay_rate * epoch)
            elif self.lr_decay_method == "step":
                return self.base_learning_rate * (self.decay_rate ** (epoch // self.decay_steps))
            else:
                raise ValueError("Not a valid decay method.")
        return self.base_learning_rate
    
    def update(self, layer: Layer, current_learning_rate: float):
        """Apply gradient updates to layer weights/biases."""
        raise NotImplementedError
    

class SGD(Optimizer):
    """Simple SGD update."""
    
    def update(self, layer: Layer, current_learning_rate: float):
        if isinstance(layer, Dense):
            layer.weight -= current_learning_rate * layer.dW
            layer.bias -= current_learning_rate * layer.dB
        elif isinstance(layer, Convolutional):
            layer.kernel -= current_learning_rate * layer.dK
            layer.bias -= current_learning_rate * layer.dB

class SGD_momentum(Optimizer):
    """SGD with momentum."""
    
    def __init__(self, learning_rate: float, momentum_rate: float = 0.9, decay_rate: float = 0.0, decay_steps: int = 1):
        super().__init__(learning_rate, decay_rate, decay_steps)
        self.momentum_rate = momentum_rate
    
    def update(self, layer: Layer, current_learning_rate: float):
        if isinstance(layer, Dense):
            # Update weights with momentum
            layer.weight -= current_learning_rate * layer.dW + self.momentum_rate * layer.last_variation
            layer.bias -= current_learning_rate * layer.dB
            
            # Store current variation for next iteration
            layer.last_variation = current_learning_rate * layer.dW
        elif isinstance(layer, Convolutional):
            layer.kernel -= current_learning_rate * layer.dK + self.momentum_rate * layer.last_variation
            layer.bias -= current_learning_rate * layer.dB
            
            # Store current variation for next iteration
            layer.last_variation = current_learning_rate * layer.dK