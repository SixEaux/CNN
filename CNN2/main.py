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


from Testing import Testing
from Training import Training
from Loss import Loss
from Activation import Activation
from Model import Model
from Dense import Dense

lr = 0.001
layers = [Dense(32, lr), Activation("sigmoid"), Dense(10, lr)]

model = Model(layers, Loss("CEL"), "mnist")

test = Testing("mnist", model)

train = Training("mnist", model, test)

print(test.exam())

train.training_simple(10)

train.plot_smthg(train.losses)
train.plot_smthg(train.accuracies)

print(test.exam())
