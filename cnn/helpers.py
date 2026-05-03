from collections.abc import Callable

def conversation_save(func:Callable, save_to:str, minus_y:bool, type_thing:str="thing"):
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

from cnn.flattening import Flattening
from cnn.convolution import Convolutional
from cnn.pooling import MaxPool
from cnn.dropout import Dropout
from cnn.activation import Activation
from cnn.dense import Dense


def load_config(name):
    ROOT = Path(__file__).resolve().parents[1]
    try:
        with open(ROOT / "config" / name, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    except FileNotFoundError:
        print("Error: config.yaml file not found")
        return None


def get_layer(layer_config: str):
    """
    Get initialized layer from name of the layer
    """
    layer_type = layer_config["type"]
    del layer_config["type"]
    if layer_type == "Convolutional":
        return Convolutional(**layer_config)
    elif layer_type == "Activation":
        return Activation(**layer_config)
    elif layer_type == "MaxPool":
        return MaxPool(**layer_config)
    elif layer_type == "Flattening":
        return Flattening(**layer_config)
    elif layer_type == "Dense":
        return Dense(**layer_config)
    elif layer_type == "Dropout":
        return Dropout(**layer_config)
    else:
        raise ValueError("Layer type not found")

