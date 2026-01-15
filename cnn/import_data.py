import pickle
import numpy as np






def split(paths:dict, validation_part:float):
    """Split data in training, testing and validation.

    Args:
        paths (dict): paths to the files
        validation_part (float): part of training to use as validation

    Returns:
        tuple: training, testing and validation sets
    """

    with open(paths["train_images"], "rb") as f:
        train_images = pickle.load(f) 

    with open(paths["train_values"], "rb") as f:
        train_values = pickle.load(f) 

    with open(paths["test_images"], "rb") as f:
        test_images = pickle.load(f) 

    with open(paths["test_values"], "rb") as f:
        test_values = pickle.load(f) 

    #randomize order images for training
    perm_train = np.random.permutation(train_images.shape[0]) 
    perm_train_images, perm_train_values = train_images[perm_train], train_values[perm_train]

    length_validation = int(train_images.shape[0]*validation_part) #get the length of the validation set
    
    return perm_train_images[length_validation:].astype(np.float32), perm_train_values[length_validation:], perm_train_images[:length_validation], perm_train_values[:length_validation], test_images.astype(np.float32), test_values


def import_data(name:str, validation_part:float=0):
    """Import the data based on which one used.

    Args:
        name (str): name of the dataset used. In ["mnist", ...]
        validation_part (float): part of training to use as validation
    """

    if name == "mnist":
        paths = {"train_images":"data/Mnist/mnist_train_images", "train_values":"data/Mnist/mnist_train_values", 
                 "test_images":"data/Mnist/mnist_test_images", "test_values":"data/Mnist/mnist_test_values"}
        labels = {i:str(i) for i in range(10)}

        train_images, train_values, validation_images, validation_values, test_images, test_values = split(paths, validation_part)
        # (60000 x 28 x 28 x 1), (60000 x 1), (10000 x 28 x 28 x 1), (10000 x 1)
        
        return train_images, train_values, validation_images, validation_values, test_images, test_values, labels
    
    elif name == "fashion_mnist":
        paths = {"train_images":"data/fashion_MNIST/fashion_train_images", "train_values":"data/fashion_MNIST/fashion_train_values", 
                 "test_images":"data/fashion_MNIST/fashion_test_images", "test_values":"data/fashion_MNIST/fashion_test_values"}
        labels = {0: "T-shirt/top", 1: "Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}
        
        train_images, train_values, validation_images, validation_values, test_images, test_values = split(paths, validation_part)
        # (60000 x 28 x 28 x 1), (60000 x 1), (10000 x 28 x 28 x 1), (10000 x 1)

        return train_images, train_values, validation_images, validation_values, test_images, test_values, labels

    else:
        raise ValueError("Not a known dataset.")
    

"""

ti, tv, qi, qv = import_data("mnist")

printgray(ti[0])
print(tv[0])

printgray(qi[0])
print(qv[0])

"""
