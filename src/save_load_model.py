import os

import numpy as np
import yaml

from src.layer import Layer

FOLDER_PATH = "outputs/trained_models"

def save_model(layers: list[Layer], config: dict, filename: str):
    """Save model's weights and architecture.

    Args:
        layers (list[Layer]): list of layers classes
        config (dict): dict extracted from yaml config file
        filename (str): name of the file where the model will be saved
    """

    if filename == "":
        print("Error: filename is empty, model not saved")
        return
    
    os.makedirs(FOLDER_PATH, exist_ok=True)

    file_parameters = f"{FOLDER_PATH}/{filename}.npz"
    file_config = f"{FOLDER_PATH}/{filename}.yaml"

    parameters = {}

    for i, layer in enumerate(layers):
        if hasattr(layer, "weight"):
            parameters[f"layer_{i}_weight"] = layer.weight
        if hasattr(layer, "bias"):
            parameters[f"layer_{i}_bias"] = layer.bias

    np.savez_compressed(file_parameters, **parameters)

    config["initialized"] = True
    config["save_name"] = filename

    with open(file_config, "w") as f:
        yaml.dump(config, f)

def load_model(filename: str):
    """Load a model trained.

    Args:
        filename (str): name of the file
    """

    with open(f"{FOLDER_PATH}/{filename}.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    parameters = np.load(f"{FOLDER_PATH}/{config['save_name']}.npz")

    return parameters, config