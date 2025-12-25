from Import_data2 import import_data
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from Model import Model
from Testing import Testing

class Training:
    """Training a model.

        Args:
            dataset (str): dataset used
            model (Model): model to train
            testing (Testing): object for testing while training
            normalize (str, optional): type of normalization. Defaults to "division".
        """

    def __init__(self, dataset: str, model: Model, testing: Testing, normalize: str = "division"):
        self.dataset = dataset
        self.training_images, self.training_values, _, _ = import_data(
            self.dataset) # import data needed
        self.model = model
        self.testing = testing

        self.losses = [] # to keep track of the losses of each iteration
        self.accuracies = [] # to keep track of the accuracy of each iteration

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

    def training_simple(self, epoch=5):
        """Training the model without batches.

        Args:
            epoch (int, optional): number of iterations through the dataset. Defaults to 5.
        """
        for e in trange(epoch, desc="Epochs"):
            mean_losses = [] # keep track of the losses of each image
            for p in trange(self.training_images.shape[0], desc="Images"):
                image = self.training_images[p].reshape(-1, 1)

                mean_losses.append(self.model.forward(
                    image, self.training_values[p])) # add the loss

                self.model.backward()

            self.losses.append(np.mean(mean_losses)) # add the mean of the losses of this iteration
            accuracy = self.testing.exam()
            print("\n", accuracy, "\n")
            self.accuracies.append(accuracy) # add the accuracy after this iteration


    def plot_smthg(self, smthg, title="", x_title="", y_title=""):
        plt.figure(figsize=(10, 6))
        plt.plot(smthg)
        plt.title(title)
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Loss")
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()
