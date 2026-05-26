import os
import pickle

from src.training import Training
from src.testing import Testing
from src.model import Model

from src.activation import Activation
from src.dense import Dense
from src.flattening import Flattening
from src.convolution import Convolutional
from src.pooling import MaxPool
from src.loss import Loss

def load_layers(architecture:list, parameters_layers:list):
    """From the list of layers names get the list of layers objects.

    Args:
        architecture (list): list with the names of layers
        parameters_layers (list): list with the parameters of each layer

    Raises:
        ValueError: I the name of the layer is not known

    Returns:
        list: objects of layers in order
    """

    layers = []

    for l in range(len(architecture)):

        if architecture[l] == "Dense":
            param = parameters_layers[l]
            new_layer = Dense(param[0].shape[0])
            new_layer.weight = param[0]
            new_layer.bias = param[1]

            layers.append(new_layer)

        elif architecture[l] == "Convolutional":
            param = parameters_layers[l]
            new_layer = Convolutional(param[0].shape[3], param[2], param[3], param[4])
            new_layer.weight = param[0]
            new_layer.bias = param[1]
            new_layer.out_dim = param[5]

            layers.append(new_layer)
        
        elif architecture[l] == "Activation":
            param = parameters_layers[l]
            new_layer = Activation(param)

            layers.append(new_layer)
        
        elif architecture[l] == "MaxPool":
            param = parameters_layers[l]
            new_layer = MaxPool(param[0], param[1])
            new_layer.out_dim = param[2]

            layers.append(new_layer)

        elif architecture[l] == "Flattening":
            param = parameters_layers[l]
            new_layer = Flattening()

            layers.append(new_layer)
        
        else:
            raise ValueError(f"I don't know this layer: {architecture[l]}")

    return layers

def load_model(filename:str):
    """Load a model trained.

    Args:
        filename (str): file where the model has been saved 

    Returns:
        Model | (Model & Training & Testing): if it is a checkpoint training load the 3 objects needed 
        and if it is not a checkpoint for training only load the model
    """
    try:
        with open(os.path.join("outputs/", "trained_models/" + filename), "rb") as f:
            saved = pickle.load(f)
        print("Loaded.")
    except FileNotFoundError:
        print("Couldn't load it.")

    layers = load_layers(saved["architecture"], saved["layers"])
    loss = Loss(saved["loss"])
    dataset = saved["dataset"]

    #create classes

    model = Model(layers, loss, dataset, initialized=True)
    model.input_size = saved["input_size"]

    if not saved["checkpoint"]:
        return model
    else:
        test = Testing(dataset, model)
        train = Training(dataset, model, test, saved["learning_rate"], lr_decay=saved["lr_decay"], lambda_rate=saved["lambda_rate"], momentum_rate=saved["momentum_rate"])
        train.finished_epochs = saved["finished_epochs"]
        train.early_stop = saved["early_stop"]
        train.losses, train.accuracies, train.validation_exams, train.validation_losses = saved["losses"], saved["accuracies"], saved["validation_accuracy"], saved["validation_losses"]
        return model, train, test
