# from cnn.training import Training
# from cnn.testing import Testing
# from cnn.loss import Loss
# from cnn.activation import Activation
# from cnn.dense import Dense
# from cnn.model import Model
# from cnn.flattening import Flattening
# from cnn.convolution import Convolutional
# from cnn.pooling import MaxPool
# from cnn.dropout import Dropout

# from cnn.save_model import save_model
# from cnn.load_model import load_model

# from cnn.visualize import visual_image, visual_outputs, confusion_matrix_plot, plot_smthg

# import numpy as np

import yaml
from pathlib import Path

def load_config():
    ROOT = Path(__file__).resolve().parents[1]
    try:
        with open(ROOT / "cnn" / "config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    except FileNotFoundError:
        print("Error: config.yaml file not found")
        return None
    

