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

from cnn.save_model import save_model

# =========================
# Hyperparameters
# =========================
lr = 0.3
dataset = "mnist"
epochs = 10
batch_size = 64

# =========================
# Model definition
# =========================
layers = [
    Convolutional(8, 3, padding=1, stride=1, initialization="he"),
    Activation("relu"),
    MaxPool(2, stride=2),

    Convolutional(16, 3, padding=1, stride=1, initialization="he"),
    Activation("relu"),
    MaxPool(2, stride=2),

    Flattening(),
    Dense(64, initialization="he"),
    Activation("relu"),
    Dense(10)
]

model = Model(
    layers=layers,
    loss=Loss("CEL"),
    dataset=dataset
)

# =========================
# Testing & Training setup
# =========================
test = Testing(dataset, model)
trainer = Training(
    dataset=dataset,
    model=model,
    testing=test,
    learning_rate=lr,
    lr_decay="exponential",
    lambda_rate=0.005,
    validation_part=1/6,
    early_stop=True
)

# =========================
# Tests
# =========================
print("=" * 50)
print("Initial evaluation")
initial_acc = test.exam()[0]
print(f"Accuracy: {initial_acc:.4f}")

print("=" * 50)
print("Training...")
trainer.SGD(epochs, batch_size, "main_model")

print("=" * 50)
print("Evaluation after training")
final_acc = test.exam()[0]
print(f"Accuracy: {final_acc:.4f}")

save_model(model, trainer, "main_model", True)

trainer.plot_smthg(trainer.losses, title="loss", x_title="Epochs", y_title="Loss", save_to="main_model")
trainer.plot_smthg(trainer.accuracies, title="accuracy", x_title="Epochs", y_title="Accuracy", save_to="main_model")
trainer.plot_smthg(trainer.learning_rates, title="learning_rates", x_title="Epochs", y_title="Learning rate", save_to="main_model")
trainer.plot_smthg(trainer.validation_losses[2:], title="validation_losses", x_title="Epochs", y_title="Loss", save_to="main_model")
trainer.plot_smthg(trainer.validation_exams, title="validation_accuracy", x_title="Epochs", y_title="Accuracy", save_to="main_model")

