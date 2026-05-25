from src.layer import Layer
import numpy as np

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
        if hasattr(layer, 'weight') and hasattr(layer, 'dW'):
            layer.weight -= current_learning_rate * layer.dW
        if hasattr(layer, 'bias') and hasattr(layer, 'dB'):
            layer.bias -= current_learning_rate * layer.dB

class SGD_momentum(Optimizer):
    """SGD with momentum."""
    
    def __init__(self, learning_rate: float, momentum_rate: float = 0.9, decay_rate: float = 0.0, decay_steps: int = 1):
        super().__init__(learning_rate, decay_rate, decay_steps)
        self.momentum_rate = momentum_rate
        self.velocity = {}  # to store momentum for each layer
    
    def update(self, layer: Layer, current_learning_rate: float):
        id_layer = id(layer)
        
        # Initialize velocities if not present
        if id_layer not in self.velocity:
            self.velocity[id_layer] = {}
            if hasattr(layer, 'weight'):
                self.velocity[id_layer]["w"] = np.zeros_like(layer.weight)
            if hasattr(layer, 'bias'):
                self.velocity[id_layer]["b"] = np.zeros_like(layer.bias)
        
        # Update weights with momentum
        if hasattr(layer, 'weight') and hasattr(layer, 'dW'):
            self.velocity[id_layer]["w"] = self.momentum_rate * self.velocity[id_layer]["w"] + layer.dW
            layer.weight -= current_learning_rate * self.velocity[id_layer]["w"]
        
        # Update bias with momentum
        if hasattr(layer, 'bias') and hasattr(layer, 'dB'):
            self.velocity[id_layer]["b"] = self.momentum_rate * self.velocity[id_layer]["b"] + layer.dB
            layer.bias -= current_learning_rate * self.velocity[id_layer]["b"]

class Adagrad(Optimizer):
    def __init__(self, learning_rate: float, decay_rate: float = 0.0, decay_steps: int = 1):
        super().__init__(learning_rate, decay_rate, decay_steps)
        self.epsilon = 1e-8
        self.cache = {}

    def get_learning_rate(self, layer: Layer):
        return self.base_learning_rate

    def update(self, layer: Layer, current_learning_rate: float):
        id_layer = id(layer)

        if id_layer not in self.cache:
            self.cache[id_layer] = {}
            if hasattr(layer, 'weight'):
                self.cache[id_layer]["w"] = np.zeros_like(layer.weight)
            if hasattr(layer, 'bias'):
                self.cache[id_layer]["b"] = np.zeros_like(layer.bias)

        if hasattr(layer, 'weight') and hasattr(layer, 'dW'):
            self.cache[id_layer]["w"] += layer.dW ** 2
            adjusted_lr_w = current_learning_rate / (np.sqrt(self.cache[id_layer]["w"]) + self.epsilon)
            layer.weight -= adjusted_lr_w * layer.dW

        if hasattr(layer, 'bias') and hasattr(layer, 'dB'):
            self.cache[id_layer]["b"] += layer.dB ** 2
            adjusted_lr_b = current_learning_rate / (np.sqrt(self.cache[id_layer]["b"]) + self.epsilon)
            layer.bias -= adjusted_lr_b * layer.dB
