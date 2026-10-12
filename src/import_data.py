import pickle
from pathlib import Path
from collections import namedtuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

# Named tuples for cleaner return values
RawDataset = namedtuple("RawDataset", ["train_images", "train_values", "test_images", "test_values"])
SplitDataset = namedtuple(
    "SplitDataset",
    ["train_images", "train_values", "validation_images", "validation_values", "test_images", "test_values"],
)
Dataset = namedtuple(
    "Dataset",
    [
        "train_images",
        "train_values",
        "validation_images",
        "validation_values",
        "test_images",
        "test_values",
        "labels",
    ],
)

DATASETS = {
    "mnist": {
        "file": "mnist.npz",
        "labels": {i: str(i) for i in range(10)},
    },
    "fashion_mnist": {
        "file": "fashion_mnist.npz",
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


def load_from_npz(path: Path, normalization_method: str):
    """Load and normalize the dataset.

    Args:
        path (Path): path to the dataset file
        normalization_method (str): type of normalization between: division, center-reduction

    Raises:
        ValueError: if the type of normalization is not known
    """
    data = np.load(path, allow_pickle=True)
    train_images = data["train_images"]
    train_values = data["train_values"]
    test_images = data["test_images"]
    test_values = data["test_values"]

    mean = np.mean(train_images)
    std = np.std(train_images)

    if normalization_method == "":
        pass
    elif normalization_method == "division":
        train_images = train_images / 255
        test_images = test_images / 255
    elif normalization_method == "center-reduction":
        train_images = (train_images - mean) / std if std != 0 else train_images - mean
        test_images = (test_images - mean) / std if std != 0 else test_images - mean
    else:
        raise ValueError("Type of normalization not known.")

    return RawDataset(train_images, train_values, test_images, test_values)


def split(path: Path, validation_part: float, normalization_method: str):
    """Split data in training, testing and validation.

    Args:
        path (Path): path to the dataset file
        validation_part (float): part of training to use as validation
        normalization_method (str): type of normalization between: division, center-reduction

    Returns:
        SplitDataset: training, testing and validation sets
    """

    raw_data = load_from_npz(path, normalization_method)
    train_images, train_values, test_images, test_values = raw_data

    perm_train = np.random.permutation(train_images.shape[0])
    perm_train_images, perm_train_values = train_images[perm_train], train_values[perm_train]

    if validation_part < 0 or validation_part >= 1:
        raise ValueError("validation_part must be between 0 and 1")

    length_validation = int(train_images.shape[0] * validation_part)

    return SplitDataset(
        perm_train_images[length_validation:].astype(np.float32),
        perm_train_values[length_validation:],
        perm_train_images[:length_validation].astype(np.float32),
        perm_train_values[:length_validation],
        test_images.astype(np.float32),
        test_values,
    )


def import_data(
    name: str,
    validation_part: float = 0,
    data_root: Path | str | None = None,
    normalization_method: str = "division",
):
    """Import the data based on which dataset is used.

    Args:
        name (str): name of the dataset used.
        validation_part (float): part of training to use as validation.
        data_root (Path | str | None): root directory containing the `data/` folder.
        normalization_method (str): type of normalization between: division, center-reduction.

    Returns:
        Dataset: train_images, train_values, validation_images, validation_values, test_images, test_values, labels
    """

    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)

    dataset = DATASETS.get(name)
    if dataset is None:
        raise ValueError(f"Unknown dataset: {name}")

    file = data_root / dataset["file"]
    if not file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {file}")

    split_data = split(file, validation_part, normalization_method=normalization_method)

    return Dataset(
        split_data.train_images.astype(np.float32),
        split_data.train_values,
        split_data.validation_images.astype(np.float32),
        split_data.validation_values,
        split_data.test_images.astype(np.float32),
        split_data.test_values,
        dataset["labels"],
    )
