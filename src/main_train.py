from src.training import Training
from src.testing import Testing
from src.loss import Loss
from src.model import Model

from src.import_data import import_data
from src.save_load_model import save_model
from src.visualize import (
    visual_image,
    visual_outputs,
    confusion_matrix_plot,
    plot_smthg,
)
from src.helpers import load_config, get_layer, get_optimizer

# =========================
# Model definition
# =========================

file_config = "config_mnist.yaml"

config = load_config(file_config)

layers = []
layers_config = config["model"]["layers"]
for i in range(len(layers_config)):
    layers.append(get_layer(layers_config[i]))

optimizer = get_optimizer(config["training"]["optimizer"])

loaded_data = import_data(
    config["dataset"],
    config["training"]["validation_part"],
    normalization_method=config["normalization_method"],
)  # import data needed

model = Model(
    layers=layers,
    loss=Loss(config["model"]["loss"], config["nb_classes"]),
    dataset=config["dataset"],
    loaded_data=loaded_data,
)

test = Testing(config["dataset"], model, loaded_data)

real_training_config = {
    k: v for k, v in config["training"].items() if (k not in ["", "epochs", "batch_size", "optimizer"])
}

trainer = Training(
    config["dataset"],
    model,
    test,
    loaded_data=loaded_data,
    optimus=optimizer,
    config=config,
    **real_training_config,
)

# =========================
# Training
# =========================
print("=" * 50)
print("Initial evaluation")
initial_acc = test.exam()[0]
print(f"Accuracy: {initial_acc:.4f}")

print("=" * 50)
print("Training...")
trainer.train(
    config["training"]["epochs"],
    config["training"]["batch_size"],
)

# =========================
# Testing
# =========================

print("=" * 50)
print("Evaluation after training")
final_acc = test.exam()[0]
print(f"Accuracy: {final_acc:.4f}")

show = config["visualization"]["show_plots"]
minus_y = config["visualization"]["minus_y"]
plot_smthg(
    trainer.losses,
    title="loss",
    x_title="Epochs",
    y_title="Loss",
    show=show,
    save_to=config["file_save"],
    minus_y=minus_y,
)
plot_smthg(
    trainer.accuracies,
    title="accuracy",
    x_title="Epochs",
    y_title="Accuracy",
    show=show,
    save_to=config["file_save"],
    minus_y=minus_y,
)
plot_smthg(
    trainer.validation_losses[2:],
    title="validation_losses",
    x_title="Epochs",
    y_title="Loss",
    show=show,
    save_to=config["file_save"],
    minus_y=minus_y,
)
plot_smthg(
    trainer.validation_exams,
    title="validation_accuracy",
    x_title="Epochs",
    y_title="Accuracy",
    show=show,
    save_to=config["file_save"],
    minus_y=minus_y,
)
save_model(model.layers, config, config["file_save"])
