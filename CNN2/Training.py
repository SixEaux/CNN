from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

from CNN2.Import_data2 import import_data
from CNN2.Model import Model
from CNN2.Testing import Testing


class Training:
    """Training a model.

        Args:
            dataset (str): dataset used
            model (Model): model to train
            testing (Testing): object for testing while training
            normalize (str, optional): type of normalization. Defaults to "division".
        """

    def __init__(self, dataset: str, model: Model, testing: Testing, normalize: str = "division", batch_size:int = 1):
        self.dataset = dataset
        self.training_images, self.training_values, _, _ = import_data(
            self.dataset)  # import data needed
        self.batch_size = batch_size
        self.model = model
        self.testing = testing

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
            self.training_images = (self.training_images - np.mean(self.training_images)) / np.std(self.training_images)
        else:
            raise ValueError("Type of normalization not known.")

    def SGD(self, epoch=5):
        """Training the model wiht or without batches (if no batches self.batch_size = 1).

        Args:
            epoch (int, optional): number of iterations through the dataset. Defaults to 5.
        """
        for e in trange(epoch, desc="Epochs"):
            mean_losses = []  # keep track of the losses of each image
            num_batches = (self.training_images.shape[0] // self.batch_size)
            for batch in range(num_batches):
                x_batch = self.training_images[batch*self.batch_size:(batch+1)*self.batch_size]
                exp_batch = self.training_values[batch*self.batch_size:(batch+1)*self.batch_size]

                mean_losses.append(self.model.forward(x_batch, exp_batch))  # add the loss

                self.model.backward(self.batch_size)
            
            # add the mean of the losses of this iteration
            self.losses.append(np.mean(mean_losses))
            accuracy = self.testing.exam()
            # add the accuracy after this iteration
            self.accuracies.append(accuracy)

    def plot_smthg(self, smthg, title="", x_title="", y_title=""):
        plt.figure(figsize=(10, 6))
        plt.plot(smthg)
        plt.title(title)
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Loss")
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()
