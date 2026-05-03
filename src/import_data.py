import pickle
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

DATASETS = {
    "mnist": {
        "folder": "Mnist",
        "files": {
            "train_images": "mnist_train_images",
            "train_values": "mnist_train_values",
            "test_images": "mnist_test_images",
            "test_values": "mnist_test_values",
        },
        "labels": {i: str(i) for i in range(10)},
    },
    "fashion_mnist": {
        "folder": "fashion_MNIST",
        "files": {
            "train_images": "fashion_train_images",
            "train_values": "fashion_train_values",
            "test_images": "fashion_test_images",
            "test_values": "fashion_test_values",
        },
        "labels": {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot",
        },
    },
}


def split(paths: dict[str, Path], validation_part: float):
    """Split data in training, testing and validation.

    Args:
        paths (dict[str, Path]): paths to the files
        validation_part (float): part of training to use as validation

    Returns:
        tuple: training, testing and validation sets
    """

    with paths["train_images"].open("rb") as f:
        train_images = pickle.load(f)

    with paths["train_values"].open("rb") as f:
        train_values = pickle.load(f)

    with paths["test_images"].open("rb") as f:
        test_images = pickle.load(f)

    with paths["test_values"].open("rb") as f:
        test_values = pickle.load(f)

    perm_train = np.random.permutation(train_images.shape[0])
    perm_train_images, perm_train_values = train_images[perm_train], train_values[perm_train]

    if validation_part < 0 or validation_part >= 1:
        raise ValueError("validation_part must be between 0 and 1")

    length_validation = int(train_images.shape[0] * validation_part)

    return (
        perm_train_images[length_validation:].astype(np.float32),
        perm_train_values[length_validation:],
        perm_train_images[:length_validation].astype(np.float32),
        perm_train_values[:length_validation],
        test_images.astype(np.float32),
        test_values,
    )


def import_data(name: str, validation_part: float = 0, data_root: Path | str | None = None):
    """Import the data based on which dataset is used.

    Args:
        name (str): name of the dataset used.
        validation_part (float): part of training to use as validation.
        data_root (Path | str | None): root directory containing the `data/` folder.

    Returns:
        tuple: train_images, train_values, validation_images, validation_values, test_images, test_values, labels
    """

    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)

    dataset = DATASETS.get(name)
    if dataset is None:
        raise ValueError(f"Unknown dataset: {name}")

    folder = data_root / dataset["folder"]
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

    paths = {key: folder / filename for key, filename in dataset["files"].items()}

    train_images, train_values, validation_images, validation_values, test_images, test_values = split(
        paths, validation_part
    )

    return (
        train_images,
        train_values,
        validation_images,
        validation_values,
        test_images,
        test_values,
        dataset["labels"],
    )
