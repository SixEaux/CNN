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


from Model import Model
from Dense import Dense
from Activation import Activation
from Loss import Loss
from Training import Training
from Testing import Testing

lr = 0.1
layers = [Dense(20, lr), Activation("sigmoid"), Dense(10, lr)]

model = Model(layers, Loss("CEL"), "mnist")

train = Training("mnist", model)

test = Testing("mnist", model)

train.training_simple(5)
print(test.exam())