import numpy as np
import yaml

from src.layer import Layer


def save_model(layers: list[Layer], config: dict, filename: str):
    """Save model's weights and architecture.

    Args:
        layers (list[Layer]): list of layers classes
        config (dict): dict extracted from yaml config file
        filename (str): name of the file where the model will be saved
    """
    file_parameters = f"outputs/trained_models/{filename}.npz"
    file_config = f"outputs/trained_models/{filename}.yaml"

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

    with open(f"outputs/trained_models/{filename}.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    parameters = np.load(config["save_name"])

    return parameters, config