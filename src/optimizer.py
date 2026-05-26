from src.layer import Layer
import numpy as np


class Optimizer:
    """Base class for optimizers."""

    def __init__(
        self,
        learning_rate: float,
        decay_rate: float = 0.0,
        decay_steps: int = 1,
        lr_decay_method: str = "exponential",
    ):
        self.base_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.lr_decay_method = lr_decay_method

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
        if hasattr(layer, "weight"):
            layer.weight -= current_learning_rate * layer.dW
            layer.bias -= current_learning_rate * layer.dB


class SGD_momentum(Optimizer):
    """SGD with momentum."""

    def __init__(
        self,
        learning_rate: float,
        momentum_rate: float = 0.9,
        decay_rate: float = 0.0,
        decay_steps: int = 1,
    ):
        super().__init__(learning_rate, decay_rate, decay_steps)
        self.momentum_rate = momentum_rate
        self.momentum_cache = {}  # to store momentum for each layer

    def update(self, layer: Layer, current_learning_rate: float):
        if not hasattr(layer, "weight"):
            return  # Skip layers without weights

        id_layer = id(layer)

        if id_layer not in self.momentum_cache:
            self.momentum_cache[id_layer] = {}
            self.momentum_cache[id_layer]["w"] = np.zeros_like(layer.weight)
            self.momentum_cache[id_layer]["b"] = np.zeros_like(layer.bias)

        if "k" not in self.momentum_cache[id_layer]:
            self.momentum_cache[id_layer]["k"] = 0

        self.momentum_cache[id_layer]["k"] += 1

        self.momentum_cache[id_layer]["w"] = (
            self.momentum_rate * self.momentum_cache[id_layer]["w"] + (1 - self.momentum_rate) * layer.dW
        )

        self.momentum_cache[id_layer]["b"] = (
            self.momentum_rate * self.momentum_cache[id_layer]["b"] + (1 - self.momentum_rate) * layer.dB
        ) 

        adjusted_momentum_w = self.momentum_cache[id_layer]["w"] / (
            1 - self.momentum_rate ** self.momentum_cache[id_layer]["k"]
        )
        adjusted_momentum_b = self.momentum_cache[id_layer]["b"] / (
            1 - self.momentum_rate ** self.momentum_cache[id_layer]["k"]
        )

        layer.weight -= current_learning_rate * adjusted_momentum_w
        layer.bias -= current_learning_rate * adjusted_momentum_b


class Adagrad(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__(learning_rate, 0, 1000)
        self.epsilon = 1e-8
        self.velocity_cache = {}

    def update(self, layer: Layer, current_learning_rate: float):
        if not hasattr(layer, "weight"):
            return  # Skip layers without weights

        id_layer = id(layer)

        if id_layer not in self.velocity_cache:
            self.velocity_cache[id_layer] = {}
            self.velocity_cache[id_layer]["w"] = np.zeros_like(layer.weight)
            self.velocity_cache[id_layer]["b"] = np.zeros_like(layer.bias)

        self.velocity_cache[id_layer]["w"] += layer.dW**2

        adjusted_lr_w = current_learning_rate / (np.sqrt(self.velocity_cache[id_layer]["w"]) + self.epsilon)

        self.velocity_cache[id_layer]["b"] += layer.dB**2

        adjusted_lr_b = current_learning_rate / (np.sqrt(self.velocity_cache[id_layer]["b"]) + self.epsilon)

        layer.weight -= adjusted_lr_w * layer.dW
        layer.bias -= adjusted_lr_b * layer.dB


class RMSprop(Optimizer):
    def __init__(self, learning_rate: float, beta: float = 0.9):
        super().__init__(learning_rate, 0, 1000)
        self.epsilon = 1e-8
        self.velocity_rate = beta
        self.velocity_cache = {}

    def update(self, layer: Layer, current_learning_rate: float):
        if not hasattr(layer, "weight"):
            return  # Skip layers without weights

        id_layer = id(layer)
        if id_layer not in self.velocity_cache:
            self.velocity_cache[id_layer] = {}
            self.velocity_cache[id_layer]["w"] = np.zeros_like(layer.weight)
            self.velocity_cache[id_layer]["b"] = np.zeros_like(layer.bias)

        if "k" not in self.velocity_cache[id_layer]:
            self.velocity_cache[id_layer]["k"] = 0

        self.velocity_cache[id_layer]["k"] += 1

        self.velocity_cache[id_layer]["w"] = (
            self.velocity_rate * self.velocity_cache[id_layer]["w"] + (1 - self.velocity_rate) * layer.dW**2
        )

        self.velocity_cache[id_layer]["b"] = (
            self.velocity_rate * self.velocity_cache[id_layer]["b"] + (1 - self.velocity_rate) * layer.dB**2
        )

        adjusted_lr_w = current_learning_rate / (
            np.sqrt(
                self.velocity_cache[id_layer]["w"]
                / (1 - self.velocity_rate ** self.velocity_cache[id_layer]["k"])
            )
            + self.epsilon
        )

        adjusted_lr_b = current_learning_rate / (
            np.sqrt(
                self.velocity_cache[id_layer]["b"]
                / (1 - self.velocity_rate ** self.velocity_cache[id_layer]["k"])
            )
            + self.epsilon
        )

        layer.weight -= adjusted_lr_w * layer.dW
        layer.bias -= adjusted_lr_b * layer.dB


class ADAM(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        momentum_rate: float = 0.9,
        velocity_rate: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate, 0, 1000)
        self.momentum_rate = momentum_rate
        self.velocity_rate = velocity_rate
        self.epsilon = epsilon
        self.momentum_cache = {}
        self.velocity_cache = {}

    def update(self, layer: Layer, current_learning_rate: float):
        if not hasattr(layer, "weight"):
            return  # Skip layers without weights

        id_layer = id(layer)

        if id_layer not in self.momentum_cache:
            self.momentum_cache[id_layer] = {}
            self.momentum_cache[id_layer]["w"] = np.zeros_like(layer.weight)
            self.momentum_cache[id_layer]["b"] = np.zeros_like(layer.bias)

        if id_layer not in self.velocity_cache:
            self.velocity_cache[id_layer] = {}
            self.velocity_cache[id_layer]["w"] = np.zeros_like(layer.weight)
            self.velocity_cache[id_layer]["b"] = np.zeros_like(layer.bias)

        if "k" not in self.momentum_cache[id_layer]:
            self.momentum_cache[id_layer]["k"] = 0

        self.momentum_cache[id_layer]["k"] += 1

        k = self.momentum_cache[id_layer]["k"]

        # MOMENTUM
        self.momentum_cache[id_layer]["w"] = (
            self.momentum_rate * self.momentum_cache[id_layer]["w"] + (1 - self.momentum_rate) * layer.dW
        )
        self.momentum_cache[id_layer]["b"] = (
            self.momentum_rate * self.momentum_cache[id_layer]["b"] + (1 - self.momentum_rate) * layer.dB
        )

        adjusted_momentum_w = self.momentum_cache[id_layer]["w"] / (1 - self.momentum_rate**k)
        adjusted_momentum_b = self.momentum_cache[id_layer]["b"] / (1 - self.momentum_rate**k)

        # VELOCITY
        self.velocity_cache[id_layer]["w"] = (
            self.velocity_rate * self.velocity_cache[id_layer]["w"] + (1 - self.velocity_rate) * layer.dW**2
        )

        self.velocity_cache[id_layer]["b"] = (
            self.velocity_rate * self.velocity_cache[id_layer]["b"] + (1 - self.velocity_rate) * layer.dB**2
        )

        adjusted_lr_w = current_learning_rate / (
            np.sqrt(
                self.velocity_cache[id_layer]["w"]
                / (1 - self.velocity_rate ** k)
            )
            + self.epsilon
        )
        adjusted_lr_b = current_learning_rate / (
            np.sqrt(
                self.velocity_cache[id_layer]["b"]
                / (1 - self.velocity_rate ** k)
            )
            + self.epsilon
        )

        # UPDATE
        layer.weight -= adjusted_lr_w * adjusted_momentum_w
        layer.bias -= adjusted_lr_b * adjusted_momentum_b
