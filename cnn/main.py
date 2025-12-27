# TODO : I need to explore initializations (try different techniques)
# TODO : I might need to create a validation sample also
# TODO : do a little bit of lr opti

from cnn.training import Training
from cnn.testing import Testing
from cnn.loss import Loss
from cnn.activation import Activation
from cnn.dense import Dense
from cnn.model import Model
from cnn.flattening import Flattening
from cnn.convolution import Convolutional
from cnn.pooling import MaxPool, MeanPool

lr = 0.08
dataset = "mnist"

layers = [
    Convolutional(20, 2, lr),
    Activation("sigmoid"),
    MeanPool(3, 3),
    Flattening(),
    Dense(10, lr)
]

model = Model(layers, Loss("CEL"), dataset)

test = Testing(dataset, model)

trainer = Training(dataset, model, test, batch_size=1)

x_batch = trainer.training_images[0:5]
exp_batch = trainer.training_values[0:5]

print(test.exam())

# model.forward(
#     x_batch, exp_batch)

# print(test.exam())

# trainer.SGD(1)

# trainer.plot_smthg(trainer.losses, x_title="Epochs", y_title="Loss")
# trainer.plot_smthg(trainer.accuracies, x_title="Epochs", y_title="Accuracy")

# print(test.exam())
