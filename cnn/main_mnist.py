# TODO : do a little bit of lr opti
# TODO : Drawing

from cnn.training import Training
from cnn.testing import Testing
from cnn.loss import Loss
from cnn.activation import Activation
from cnn.dense import Dense
from cnn.model import Model
from cnn.flattening import Flattening
from cnn.convolution import Convolutional
from cnn.pooling import MaxPool
from cnn.dropout import Dropout

from cnn.save_model import save_model
from cnn.load_model import load_model

from cnn.visualize import visual_image, visual_outputs, confusion_matrix_plot, plot_smthg
from cnn.drawing import Draw

import numpy as np

# =========================
# Hyperparameters
# =========================
lr = 0.05
dataset = "fashion_mnist"
epochs = 20
batch_size = 64
file = "test_draw"

"""
# =========================
# Model definition
# =========================

initialization = "he"
acti_func = "relu"

layers = [
    Convolutional(8, 3, padding=1, stride=1, initialization=initialization),
    Activation(acti_func),
    MaxPool(2, stride=2),

    Convolutional(16, 3, padding=1, stride=1, initialization=initialization),
    Activation(acti_func),
    MaxPool(2, stride=2),

    Flattening(),
    Dense(64, initialization=initialization),
    Activation(acti_func),
 
    Dropout(0.3),

    Dense(10)
]

model = Model(
    layers=layers,
    loss=Loss("CEL"),
    dataset=dataset,
    CAM_image=[i for i in range(10)]
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
    lambda_rate=0.01,
    validation_part=1/6,
    early_stop=True,
    momentum_rate=0.9,
    patience=2, 
    min_epoch=5
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
trainer.SGD(epochs, batch_size) 

print("=" * 50)
print("Evaluation after training")
final_acc = test.exam()[0]
print(f"Accuracy: {final_acc:.4f}")

show = True
minus_y = False
plot_smthg(trainer.losses, title="loss", x_title="Epochs", y_title="Loss", show=show, save_to=file, minus_y=minus_y)
plot_smthg(trainer.accuracies, title="accuracy", x_title="Epochs", y_title="Accuracy", show=show, save_to=file, minus_y=minus_y)
plot_smthg(trainer.learning_rates, title="learning_rates", x_title="Epochs", y_title="Learning rate", show=show, save_to=file, minus_y=minus_y)
plot_smthg(trainer.validation_losses[2:], title="validation_losses", x_title="Epochs", y_title="Loss", show=show, save_to=file, minus_y=minus_y)
plot_smthg(trainer.validation_exams, title="validation_accuracy", x_title="Epochs", y_title="Accuracy", show=show, save_to=file, minus_y=minus_y)
save_model(model, trainer, file, True)

"""



model, trainer, test = load_model(file)

d = Draw(model, trainer, learn_bool=False)


# visual_image(trainer.training_images[:40])

# confusion_matrix_plot(t[2], test.testing_values, test.labels.values(), title="confusion_matrix_fashion_test", save_to=file, minus_y=False)
