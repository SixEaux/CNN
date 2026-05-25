from src.training import Training
from src.testing import Testing
from src.loss import Loss
from src.model import Model

from src.import_data import import_data
from src.save_model import save_model
from src.visualize import (
    visual_image,
    visual_outputs,
    confusion_matrix_plot,
    plot_smthg,
)
from src.helpers import load_config, get_layer, get_optimizer

from src.cam_image import CAM_IMAGE

# =========================
# Model definition
# =========================

config = load_config("config_mnist.yaml")

layers = []
layers_config = config["model"]["layers"]
for i in range(len(layers_config)):
    layers.append(get_layer(layers_config[i]))

optimizer = get_optimizer(config["training"]["optimizer"])

loaded_data = import_data(
    config["dataset"], config["training"]["validation_part"]
)  # import data needed

# Prepare CAM images using indices from config
_, _, test_images, _, _, _, _ = loaded_data
cam_image_indices = config["CAM_image"] if config["CAM_image"] else []
cam = CAM_IMAGE(test_images[cam_image_indices]) if cam_image_indices else None

model = Model(
    layers=layers,
    loss=Loss(config["model"]["loss"], config["nb_classes"]),
    dataset=config["dataset"],
    initialized=config["model"]["initialized"],
    cam=cam,
)

test = Testing(config["dataset"], model, loaded_data)

real_training_config = {
    k: v for k, v in config["training"].items() if (k not in ["", "epochs", "batch_size", "optimizer"])
}

trainer = Training(
    config["dataset"],
    model,
    test,
    optimus=optimizer,
    loaded_data=loaded_data,
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
    to_save=config["save_file"],
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
    save_to=config["save_file"],
    minus_y=minus_y,
)
plot_smthg(
    trainer.accuracies,
    title="accuracy",
    x_title="Epochs",
    y_title="Accuracy",
    show=show,
    save_to=config["save_file"],
    minus_y=minus_y,
)
plot_smthg(
    trainer.validation_losses[2:],
    title="validation_losses",
    x_title="Epochs",
    y_title="Loss",
    show=show,
    save_to=config["save_file"],
    minus_y=minus_y,
)
plot_smthg(
    trainer.validation_exams,
    title="validation_accuracy",
    x_title="Epochs",
    y_title="Accuracy",
    show=show,
    save_to=config["save_file"],
    minus_y=minus_y,
)
save_model(model, trainer, config["save_file"], checkpoint=False, minus_y=False)
