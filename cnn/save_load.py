import os
import pickle

from cnn.training import Training
from cnn.testing import Testing
from cnn.model import Model

from cnn.activation import Activation
from cnn.dense import Dense
from cnn.flattening import Flattening
from cnn.convolution import Convolutional
from cnn.pooling import MaxPool, MeanPool
from cnn.loss import Loss

def get_weights_architecture(layers:list):
    layers_parameters = []
    architecture = []
    for l in layers:
        if isinstance(l, Dense):
            layers_parameters.append((l.weight, l.bias))
            architecture.append("Dense")
        elif isinstance(l, Convolutional):
            layers_parameters.append((l.kernel, l.bias, l.size_kernel, l.stride, l.padding))
            architecture.append("Convolutional")
        elif isinstance(l, Activation):
            layers_parameters.append(l.function)
            architecture.append("Activation")
        elif isinstance(l, MaxPool):
            layers_parameters.append((l.size_kernel, l.stride))
            architecture.append("MaxPool")
        elif isinstance(l, MeanPool):
            layers_parameters.append((l.size_kernel, l.stride))
            architecture.append("MeanPool")
        elif isinstance(l, Flattening):
            layers_parameters.append(None)
            architecture.append("Flattening")

    return layers_parameters, architecture

def extract_model_state(layers: list, loss: Loss, train:Training):
    layers_params, architecture = get_weights_architecture(layers)
    return {
        "architecture": architecture,
        "layers": layers_params,
        "loss": loss.function,
        "dataset": train.dataset
    }

def extract_training_state(train: Training):
    return {
        "learning_rate": train.learning_rate,
        "lr_decay": train.lr_decay_method,
        "lambda_rate": train.lambda_rate,
        "momentum_rate": train.momentum_rate,
        "finished_epochs": train.finished_epochs,
    }

def extract_history(train: Training):
    return {
        "losses": train.losses,
        "accuracies": train.accuracies,
        "learning_rates": train.learning_rates,
    }

def save_model(model:Model, train:Training, filename:str, checkpoint:bool=False):
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

def load_layers(architecture, parameters_layers):
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
            new_layer.kernel = param[0]
            new_layer.bias = param[1]

            layers.append(new_layer)
        
        elif architecture[l] == "Activation":
            param = parameters_layers[l]
            new_layer = Activation(param)

            layers.append(new_layer)
        
        elif architecture[l] == "MaxPool":
            param = parameters_layers[l]
            new_layer = MaxPool(param[0], param[1])

            layers.append(new_layer)

        elif architecture[l] == "MeanPool":
            param = parameters_layers[l]
            new_layer = MeanPool(param[0], param[1])

            layers.append(new_layer)

        elif architecture[l] == "Flattening":
            param = parameters_layers[l]
            new_layer = Flattening()

            layers.append(new_layer)
        
        else:
            raise ValueError(f"I don't know this layer: {architecture[l]}")

    return layers

def load_model(filename:str):
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

    if not saved["checkpoint"]:
        return model
    else:
        test = Testing(dataset, model)
        train = Training(dataset, model, test, saved["learning_rate"], lr_decay=saved["lr_decay"], lambda_rate=saved["lambda_rate"], momentum_rate=saved["momentum_rate"])
        train.finished_epochs = saved["finished_epochs"]
        train.losses, train.accuracies, train.learning_rates = saved["losses"], saved["accuracies"], saved["learning_rates"]
        return model, train, test
