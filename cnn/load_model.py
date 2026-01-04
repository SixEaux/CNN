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
