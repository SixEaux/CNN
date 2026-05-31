from src.save_load_model import load_model

from src.training import Training
from src.testing import Testing
from src.loss import Loss
from src.model import Model

from src.import_data import import_data
from src.helpers import get_layer, get_optimizer

name_config = "test_save_load"

parameters, config = load_model(name_config)

parameters_clean = {
    f"layer_{i}": {
        "weight": parameters.get(f"layer_{i}_weight", None),
        "bias": parameters.get(f"layer_{i}_bias", None),
    }
    for i in range(len(config["model"]["layers"]))
}

layers = []
layers_config = config["model"]["layers"]

for i in range(len(layers_config)):
    layers.append(
        get_layer(
            layers_config[i],
        )
    )

optimizer = get_optimizer(config["training"]["optimizer"])

loaded_data = import_data(config["dataset"], config["training"]["validation_part"])  # import data needed

model = Model(
    layers=layers,
    loss=Loss(config["model"]["loss"], config["nb_classes"]),
    dataset=config["dataset"],
    initialized=config["initialized"],
)

for i in range(len(layers_config)):
    if parameters_clean[f"layer_{i}"]["weight"] is not None:
        model.layers[i].weight = parameters_clean[f"layer_{i}"]["weight"]
    if parameters_clean[f"layer_{i}"]["bias"] is not None:
        model.layers[i].bias = parameters_clean[f"layer_{i}"]["bias"]


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
    **real_training_config,
)

# =========================
# Testing
# =========================

print("=" * 50)
print("Evaluation after training")
final_acc = test.exam()[0]
print(f"Accuracy: {final_acc:.4f}")
