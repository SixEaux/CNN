"""
To facilitate passing parameters -> to create a model you need to:
    Model(
    layers = [Convolutional(), Activation(), Flatening(), Pooling(), Dense()], 
    other_parameters)
"""

"""
As a convention when i use:
- x it is an input
- C or L cost / loss
- w weights
- b biais
- z = wx + b
- a = activ(z)
"""

# TODO : Batch gradient descent
# TODO : I need to explore initializations (try different techniques)
# TODO : I might need to create a validation sample also
# TODO : maybe quickly do a little bit of lr opti

from CNN2.Model import Model
from CNN2.Flattening import Flattening
from CNN2.Dense import Dense
from CNN2.Activation import Activation
from CNN2.Loss import Loss
from CNN2.Testing import Testing
from CNN2.Training import Training

lr = 0.005
layers = [
    Flattening(),
    Dense(64, lr),
    Activation("relu"),
    Dense(10, lr)
]

model = Model(layers, Loss("CEL"), "mnist")

test = Testing("mnist", model)

trainer = Training("mnist", model, test, batch_size=32)

print(test.exam())

trainer.SGD(10) 

trainer.plot_smthg(trainer.losses, x_title="Epochs", y_title="Loss")
trainer.plot_smthg(trainer.accuracies, x_title="Epochs", y_title="Accuracy")

print(test.exam())