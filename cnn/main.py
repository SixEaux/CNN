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

from cnn.Training import Training
from cnn.Testing import Testing
from cnn.Loss import Loss
from cnn.Activation import Activation
from cnn.Dense import Dense
from cnn.Model import Model
from cnn.Flattening import Flattening

lr = 5
dataset = "mnist"

layers = [
    Flattening(),
    Dense(64, lr),
    Activation("sigmoid"),
    Dense(10, lr)
]

model = Model(layers, Loss("CEL"), dataset)

test = Testing(dataset, model)

trainer = Training(dataset, model, test, batch_size=60000)

print(test.exam())

trainer.SGD(10)

trainer.plot_smthg(trainer.losses, x_title="Epochs", y_title="Loss")
trainer.plot_smthg(trainer.accuracies, x_title="Epochs", y_title="Accuracy")

print(test.exam())
