from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import os

from cnn.import_data import import_data
from cnn.model import Model
from cnn.testing import Testing


class Training:
    """Training a model.

        Args:
            dataset (str): dataset used
            model (Model): model to train
            testing (Testing): object for testing while training
            normalize (str, optional): type of normalization. Defaults to "division".
            learning_rate (float): learning rate

            lr_decay (str): method of learning rate decay
            lambda_rate (float): lambda parameter for exponential decay and inverse decay
            momentum_rate (float): value for parameter of momentum
        """

    def __init__(self, dataset: str, model: Model, testing: Testing, learning_rate:float, normalize: str = "division", batch_size: int = 1,
                 lr_decay:str="",lambda_rate:float=0, momentum_rate:float=0):
        self.dataset = dataset
        self.training_images, self.training_values, _, _ = import_data(
            self.dataset)  # import data needed
        self.batch_size = batch_size
        self.model = model
        self.testing = testing
        self.learning_rate = learning_rate

        self.initial_lr = learning_rate
        self.lr_decay_method = lr_decay
        self.lambda_rate = lambda_rate
        self.momentum_rate = momentum_rate
        self.finished_epochs = 0
        self.learning_rates = [self.learning_rate]

        self.losses = []  # to keep track of the losses of each iteration
        self.accuracies = []  # to keep track of the accuracy of each iteration

        self.normalization(normalize)

    def normalization(self, type: str):
        """Normalize the dataset.

        Args:
            type (str): type of normalization between: division, center-reduction

        Raises:
            ValueError: if the type of normalization is not known
        """
        if type == "":
            return
        elif type == "division":
            self.training_images = self.training_images / 255
        elif type == "center-reduction":
            self.training_images = (
                self.training_images - np.mean(self.training_images)) / np.std(self.training_images)
        else:
            raise ValueError("Type of normalization not known.")

    def SGD(self, epoch: int = 1):
        """Training the model wiht or without batches (if no batches self.batch_size = 1).

        Args:   
            epoch (int, optional): number of iterations through the dataset. Defaults to 5.
        """
        for e in trange(epoch, desc="Epochs"):
            mean_losses = []  # keep track of the losses of each image
            num_batches = (self.training_images.shape[0] // self.batch_size)
            for batch in trange(num_batches, desc="Batch"):
                x_batch = self.training_images[batch *
                                               self.batch_size:(batch+1)*self.batch_size]
                exp_batch = self.training_values[batch *
                                                 self.batch_size:(batch+1)*self.batch_size]

                mean_losses.append(self.model.forward(
                    x_batch, exp_batch))  # add the loss

                self.model.backward(self.batch_size, self.learning_rate)

            self.finished_epochs += 1
            self.lr_decay()
            self.learning_rates.append(self.learning_rate)

            # add the mean of the losses of this iteration
            self.losses.append(np.mean(mean_losses))
            accuracy = self.testing.exam()
            # add the accuracy after this iteration
            self.accuracies.append(accuracy)

    def lr_decay(self):
        if self.lr_decay_method == "":
            return 
        elif self.lr_decay_method == "exponential":
            self.learning_rate = self.initial_lr * np.exp(-self.lambda_rate*self.finished_epochs)
        elif self.lr_decay_method == "inverse":
            self.learning_rate = self.initial_lr / (1 + self.lambda_rate * self.finished_epochs)
        else:
            raise ValueError("Not a valid decay method.")

    def plot_smthg(self, smthg:np.ndarray, save_to:str, title:str="", x_title:str="", y_title:str="", show:bool=False):
        plt.figure(figsize=(10, 6))
        plt.plot(smthg)
        plt.title(title)
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Loss")
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        dir_plots = os.path.join("outputs", "plots")
        new_folder_path = os.path.join(dir_plots, save_to)
        os.makedirs(new_folder_path, exist_ok=True)

        plt.savefig(os.path.join(new_folder_path, title))

        if show:
            plt.show()
