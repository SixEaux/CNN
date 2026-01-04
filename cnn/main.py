# TODO : I might need to create a validation sample also
# TODO : do a little bit of lr opti
# TODO : Visualization and drawing

from cnn.training import Training
from cnn.testing import Testing
from cnn.loss import Loss
from cnn.activation import Activation
from cnn.dense import Dense
from cnn.model import Model
from cnn.flattening import Flattening
from cnn.convolution import Convolutional
from cnn.pooling import MaxPool, MeanPool

from cnn.save_load import save_model, load_model

lr = 0.05
dataset = "mnist"

layers = [
    Convolutional(5, 4, padding=2, stride=2),
    Activation("sigmoid"),
    Convolutional(2, 2),
    Flattening(),
    Dense(10)
]

model = Model(layers, Loss("CEL"), dataset)

test = Testing(dataset, model)

trainer = Training(dataset, model, test, learning_rate=lr, lr_decay="", lambda_rate=0)

print("Initial test", test.exam())

trainer.SGD(1, batch_size=32)

print("After training test", test.exam())
