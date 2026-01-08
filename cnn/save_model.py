import os
import pickle

from cnn.activation import Activation
from cnn.dense import Dense
from cnn.flattening import Flattening
from cnn.convolution import Convolutional
from cnn.pooling import MaxPool, MeanPool
from cnn.loss import Loss

from cnn.helpers import conversation_save

def get_weights_architecture(layers:list):
    """Get layers and architecture for saving

    Args:
        layers (list): list of layers objects

    Returns:
        tuple: layers list of parameters and list of names layers
    """
    layers_parameters = []
    architecture = []
    for l in layers:
        if isinstance(l, Dense):
            layers_parameters.append((l.weight, l.bias))
            architecture.append("Dense")
        elif isinstance(l, Convolutional):
            layers_parameters.append((l.kernel, l.bias, l.size_kernel, l.stride, l.padding, l.out_dim))
            architecture.append("Convolutional")
        elif isinstance(l, Activation):
            layers_parameters.append(l.function)
            architecture.append("Activation")
        elif isinstance(l, MaxPool):
            layers_parameters.append((l.size_kernel, l.stride, l.out_dim))
            architecture.append("MaxPool")
        elif isinstance(l, MeanPool):
            layers_parameters.append((l.size_kernel, l.stride, l.out_dim))
            architecture.append("MeanPool")
        elif isinstance(l, Flattening):
            layers_parameters.append(None)
            architecture.append("Flattening")

    return layers_parameters, architecture

def extract_model_state(layers: list, loss: Loss, train):
    """Extract model state for saving.

    Args:
        layers (list): list of layers classes
        loss (Loss): loss classes
        train (Training): Training class

    Returns:
        dict: model state
    """
    layers_params, architecture = get_weights_architecture(layers)
    return {
        "architecture": architecture,
        "layers": layers_params,
        "loss": loss.function,
        "dataset": train.dataset
    }

def extract_training_state(train):
    """Extract training state.

    Args:
        train (Training): class

    Returns:
        dict: training state
    """
    return {
        "learning_rate": train.learning_rate,
        "lr_decay": train.lr_decay_method,
        "lambda_rate": train.lambda_rate,
        "momentum_rate": train.momentum_rate,
        "finished_epochs": train.finished_epochs,
        "early_stop": train.early_stop,
        }

def extract_history(train):
    """Extract history of training.

    Args:
        train (Training): class

    Returns:
        dict: history of training
    """
    return {
        "losses": train.losses,
        "accuracies": train.accuracies,
        "learning_rates": train.learning_rates,
        "validation_losses": train.validation_losses,
        "validation_accuracy": train.validation_exams,
    }

def save_model(model, train, filename:str="", checkpoint:bool=False, minus_y:bool=False):
    """Save the model.

    Args:
        model (Model): model class
        train (Training): training class
        filename (str): file to save model
        checkpoint (bool, optional): is it a checkpoint save or not (for continuing training later). Defaults to False.
        minus_y (bool): save without asking
    """

    def save():
        model_state = extract_model_state(model.layers, model.loss, train)
        if not checkpoint:
            to_save = {**model_state, "checkpoint":False}
        else:
            to_save = {**model_state, **extract_history(train), **extract_training_state(train), "checkpoint":True}
        try:
            with open(os.path.join("outputs/", "trained_models/" + filename), "wb") as f:
                pickle.dump(to_save, f)
            print("SAVED.")
        except FileNotFoundError:
            print("Couldn't save.")
    
    conversation_save(save, filename, minus_y)
        

    
    
    