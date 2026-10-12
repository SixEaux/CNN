import yaml
from pathlib import Path

from src.flattening import Flattening
from src.convolution import Convolutional
from src.pooling import MaxPool
from src.dropout import Dropout
from src.activation import Activation
from src.dense import Dense

from src.optimizer import SGD, SGD_momentum, Adagrad, RMSprop, ADAM

def load_config(name):
    ROOT = Path(__file__).resolve().parents[1]
    with open(ROOT / "config" / name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def get_layer(layer_config: str):
    """
    Get initialized layer from name of the layer
    """
    layer_type = layer_config["type"]
    layer_config_copy = layer_config.copy() 
    del layer_config_copy["type"]
    if layer_type == "Convolutional":
        layer = Convolutional(**layer_config_copy)
    elif layer_type == "Activation":
        layer = Activation(**layer_config_copy)
    elif layer_type == "MaxPool":
        layer = MaxPool(**layer_config_copy)
    elif layer_type == "Flattening":
        layer = Flattening(**layer_config_copy)
    elif layer_type == "Dense":
        layer = Dense(**layer_config_copy)
    elif layer_type == "Dropout":
        layer = Dropout(**layer_config_copy)
    else:
        raise ValueError("Layer type not found")
    
    return layer


def get_optimizer(optimizer_config: dict):
    optimizer_type = optimizer_config["type"]
    optimizer_config_copy = optimizer_config.copy()
    del optimizer_config_copy["type"]
    if optimizer_type == "SGD":
        return SGD(**optimizer_config_copy)
    elif optimizer_type == "SGD_momentum":
        return SGD_momentum(**optimizer_config_copy)
    elif optimizer_type == "Adagrad":
        return Adagrad(**optimizer_config_copy)
    elif optimizer_type == "RMSprop":
        return RMSprop(**optimizer_config_copy)
    elif optimizer_type == "Adam":
        return ADAM(**optimizer_config_copy)
    else:
        raise ValueError("Optimizer type not found: " + optimizer_type)
