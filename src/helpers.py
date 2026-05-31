from collections.abc import Callable
from src.optimizer import *


def conversation_save(func: Callable, save_to: str, minus_y: bool, type_thing: str = "thing"):
    if save_to != "":

        if minus_y:
            func()

        else:

            while True:

                i = input(f"Are you sure you want to save this {type_thing}? (y/n)")

                if i == "y":
                    func()
                    break

                elif i == "n":
                    print("You decided not to save.")
                    break

                elif i == "oh no an infinite loop":
                    print("Don't worry, I am here")
                    break

                else:
                    print("Not a valid input.")


import yaml
from pathlib import Path

from src.flattening import Flattening
from src.convolution import Convolutional
from src.pooling import MaxPool
from src.dropout import Dropout
from src.activation import Activation
from src.dense import Dense


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
