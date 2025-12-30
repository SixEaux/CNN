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

lr = 0.5
dataset = "mnist"

layers = [
    Convolutional(5, 4, stride=2),
    Activation("sigmoid"),
    MaxPool(2),
    Flattening(),
    Dense(10)
]

model = Model(layers, Loss("CEL"), dataset, lr)

test = Testing(dataset, model)

trainer = Training(dataset, model, test, batch_size=50)

print("Initial test", test.exam())

trainer.SGD(5)

print("After training test", test.exam())

trainer.plot_smthg(trainer.losses, title="test_model_loss", x_title="Epochs", y_title="Loss", save_to="test_model")
trainer.plot_smthg(trainer.accuracies, title="test_model_accuracy", x_title="Epochs", y_title="Accuracy", save_to="test_model")
