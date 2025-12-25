"""
To facilitate passing parameters -> to create a model you need to:
    Model(
    layers = [Convolutional(), Activation(), Flatening(), Pooling(), Dense()], 
    other_parameters)
"""

"""
As a convention when i use:
- x it is an input
- C cost / loss
- w weights
- b biais
- z = wx + b
- a = activ(z)
"""

# TODO : Batch gradient descent
# TODO : I need to explore initializations (try different techniques)
# TODO : I might need to create a validation sample also
# TODO : maybe quickly do a little bit of lr opti

from Testing import Testing
from Training import Training
from Loss import Loss
from Activation import Activation
from Model import Model
from Dense import Dense

lr = 0.18
layers = [Dense(100, lr), Activation("sigmoid"), Dense(50, lr), Activation("sigmoid"), Dense(10, lr)]

model = Model(layers, Loss("CEL"), "mnist")

test = Testing("mnist", model)

train = Training("mnist", model, test)

print(test.exam())

train.training_simple(20)

train.plot_smthg(train.losses, x_title="Epochs", y_title="Loss")
train.plot_smthg(train.accuracies, x_title="Epochs", y_title="Accuracy")

print(test.exam())
