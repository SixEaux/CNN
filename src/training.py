from typing import NamedTuple

from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import os

from src.import_data import import_data
from src.model import Model
from src.testing import Testing
from src.optimizer import Optimizer

from src.save_load_model import save_model
from src.helpers import conversation_save


class Training:
    """Training a model.

    Args:
        dataset (str): dataset used
        model (Model): model to train
        testing (Testing): object for testing while training
        learning_rate (float): learning rate

        lr_decay (str): method of learning rate decay
        lambda_rate (float): lambda parameter for exponential decay and inverse decay
        momentum_rate (float): value for parameter of momentum

        validation_part (float, optional): part of the training set used for validation. Defaults to 0.

        early_stop (bool, optional): stop early the training following method. Defaults to False.
        patience (int): how many epochs to wait before early stopping
        min_epoch (int): at which epoch start early stoppoing
    """

    def __init__(
        self,
        dataset: str,
        model: Model,
        testing: Testing,
        optimus: Optimizer,
        loaded_data: NamedTuple,
        validation_part: float = 0,
        early_stop: bool = False,
        patience: int = 1,
        min_epoch: int = 5,
        config: dict = None,
    ):

        self.dataset = dataset
        self.validation_part = validation_part

        self.training_images = loaded_data.train_images
        self.training_values = loaded_data.train_values
        self.validation_images = loaded_data.validation_images
        self.validation_values = loaded_data.validation_values

        self.model = model
        self.testing = testing

        self.optimizer = optimus

        self.finished_epochs = 0  # number of finished epochs
        self.losses = []  # keep track of the losses of each iteration
        self.accuracies = []  # keep track of the accuracy of each iteration
        self.validation_exams = []  # keep track of the accuracy of each iteration on validation set
        self.validation_losses = []  # keep track loss in validation sample of this iteration

        self.early_stop = early_stop
        self.patience = patience
        self.count_stop = 0
        self.min_epoch = min_epoch

        self.config = config

        if self.early_stop and self.validation_part == 0:
            raise ValueError("Early stopping cannot be used without validation set")

    def is_in(self, image: np.ndarray, batch: np.ndarray):
        """None if the image is not in the batch or the position in the batch.

        Args:
            image (ndarray): to find DIM = (h, w, c)
            batch (ndarray): batch of images DIM = (b, h, w, c)
        """

        h, w, c = image.shape

        mask = batch == image

        number_equal_pixels = np.sum(mask, axis=(1, 2, 3))

        mask_2 = number_equal_pixels == h * w * c

        if np.any(mask_2):
            return np.argmax(mask_2)
        else:
            return

    def training_epoch(self, batch_size: int, num_batches: int):
        """Do an iteration of training.

        Args:
            batch_size (int): size of the batch
            num_batches (int): number of batches that divides dataset
        """

        # permutate the data for new batches
        perm = np.random.permutation(self.training_images.shape[0])

        current_lr = self.optimizer.get_learning_rate(self.finished_epochs)

        for batch in trange(num_batches, desc="Batch"):
            batch_index = perm[batch * batch_size : (batch + 1) * batch_size]
            x_batch = self.training_images[batch_index]
            exp_batch = self.training_values[batch_index]

            self.model.forward(x_batch, exp_batch)

            self.model.backward(batch_size, batch_images=x_batch)

            for layer in self.model.layers:
                self.optimizer.update(layer, current_lr)

    def early_stopping(self):
        """For the moment just simple early stopping but in the future try various methods.

        Returns:
            bool: if True stop
        """
        if (
            self.validation_losses[-2] < self.validation_losses[-1]
        ):  # if the loss from previous iteration is smaller than this one
            return True
        else:
            return False

    def end_iteration(self):
        """Stuff to do after the iteration."""
        self.finished_epochs += 1

        # add important info of this iteration
        results = self.testing.exam()
        self.accuracies.append(results[0])
        self.losses.append(results[1])

        # add info about validation set
        if self.validation_part > 0:
            results_validation = self.testing.exam(self.validation_images, self.validation_values)
            self.validation_exams.append(results_validation[0])
            self.validation_losses.append(results_validation[1])

    def train(self, epoch: int = 1, batch_size: int = 1, file_save: str = ""):
        """Training the model wiht or without batches (if no batches batch_size = 1).

        Args:
            epoch (int, optional): number of iterations through the dataset. Defaults to 5.
            batch_size (int): size of the batch
            file_save (str): if given saves the model to the file every 10 iterations
        """
        num_batches = self.training_images.shape[0] // batch_size

        for e in trange(epoch, desc="Epochs"):

            self.training_epoch(batch_size, num_batches)

            self.end_iteration()

            if file_save != "" and e % 10 == 0:
                save_model(self.model, self, file_save, checkpoint=True, minus_y=True)

            # early stopping
            if self.early_stop and e >= self.min_epoch:
                b = self.early_stopping()
                if b:
                    self.count_stop += 1
                else:
                    self.count_stop = 0

                if self.count_stop >= self.patience:
                    if file_save != "":
                        save_model(self.model.layers, self.config, file_save)
                    print(f"Early stop at epoch {e}")
                    break
