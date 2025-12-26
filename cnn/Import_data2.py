import pickle
import numpy as np

def import_data(name:str):
    """Import the data based on which one used.

    Args:
        name (str): name of the dataset used. In ["mnist", ...]
    """

    if name == "mnist":
        paths = ["data\Mnist\mnist_train_images", "data\Mnist\mnist_train_values", "data\Mnist\mnist_test_images", "data\Mnist\mnist_test_values"]

        with open(paths[0], "rb") as f:
            train_images = pickle.load(f) # 60000 x 28 x 28 x 1

        with open(paths[1], "rb") as f:
            train_values = pickle.load(f) # 60000 x 1

        with open(paths[2], "rb") as f:
            test_images = pickle.load(f) # 10000 x 28 x 28 x 1

        with open(paths[3], "rb") as f:
            test_values = pickle.load(f) # 10000 x 1

        perm_train = np.random.permutation(train_images.shape[0])

        return train_images[perm_train], train_values[perm_train], test_images, test_values

    else:
        raise ValueError("Not a known dataset.")
    

"""

ti, tv, qi, qv = import_data("mnist")

printgray(ti[0])
print(tv[0])

printgray(qi[0])
print(qv[0])

"""
