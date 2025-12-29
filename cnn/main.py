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

lr = 0.01
dataset = "mnist"

layers = [
    Convolutional(5, 4, lr, stride=2),
    Activation("sigmoid"),
    MaxPool(2),
    Flattening(),
    Dense(10, lr)
]

model = Model(layers, Loss("CEL"), dataset)

test = Testing(dataset, model)

trainer = Training(dataset, model, test, batch_size=32)

print("Initial test", test.exam())

trainer.SGD(1)

print("After training test", test.exam())

# trainer.plot_smthg(trainer.losses, x_title="Epochs", y_title="Loss")
# trainer.plot_smthg(trainer.accuracies, x_title="Epochs", y_title="Accuracy")
