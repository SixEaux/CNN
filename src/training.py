from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import os

from cnn.import_data import import_data
from cnn.model import Model
from cnn.testing import Testing

from cnn.save_model import save_model
from cnn.helpers import conversation_save


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
        learning_rate: float,
        normalize: str = "division",
        lr_decay: str = "",
        lambda_rate: float = 0,
        momentum_rate: float = 0,
        validation_part: float = 0,
        early_stop: bool = False,
        patience: int = 1,
        min_epoch: int = 5,
        loaded_data: tuple = None,
    ):

        self.dataset = dataset
        self.validation_part = validation_part

        (
            self.training_images,
            self.training_values,
            self.validation_images,
            self.validation_values,
            _,
            _,
            _,
        ) = (
            import_data(self.dataset, validation_part) if loaded_data is None else loaded_data
        )  # import data needed 

        self.model = model
        self.testing = testing
        self.learning_rate = learning_rate

        self.initial_lr = learning_rate
        self.lr_decay_method = lr_decay
        self.lambda_rate = lambda_rate
        self.momentum_rate = momentum_rate

        self.finished_epochs = 0  # number of finished epochs
        self.losses = []  # keep track of the losses of each iteration
        self.accuracies = []  # keep track of the accuracy of each iteration
        self.validation_exams = []  # keep track of the accuracy of each iteration on validation set
        self.validation_losses = []  # keep track loss in validation sample of this iteration

        self.early_stop = early_stop
        self.patience = patience
        self.count_stop = 0
        self.min_epoch = min_epoch

        self.normalization(normalize)

        if self.model.CAM_image is not None:
            self.CAM_image = [
                self.training_images[i] if isinstance(i, int) else i for i in self.model.CAM_image
            ]  # images we want to follow in CAM
        else:
            self.CAM_image = None

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
            self.training_images = (self.training_images - np.mean(self.training_images)) / np.std(
                self.training_images
            )
        else:
            raise ValueError("Type of normalization not known.")

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

    def training_iteration(self, batch_size: int, num_batches: int):
        """Do an iteration of training.

        Args:
            batch_size (int): size of the batch
            num_batches (int): number of batches that divides dataset
        """

        save = [None for _ in range(len(self.CAM_image))] if self.CAM_image is not None else None

        for batch in trange(num_batches, desc="Batch"):
            x_batch = self.training_images[batch * batch_size : (batch + 1) * batch_size]
            exp_batch = self.training_values[batch * batch_size : (batch + 1) * batch_size]

            if self.CAM_image is not None:
                for i in range(len(self.CAM_image)):
                    save[i] = self.is_in(self.CAM_image[i], x_batch)

            self.model.forward(x_batch, exp_batch)

            self.model.backward(batch_size, self.learning_rate, self.momentum_rate, save)

            save = (
                [None for _ in range(len(self.CAM_image))] if self.CAM_image is not None else None
            )

    def lr_decay(self):
        """Decay the learning rate based on the methof used.

        Raises:
            ValueError: if the method of learning decay is unkown
        """
        if self.lr_decay_method == "exponential":
            self.learning_rate = self.initial_lr * np.exp(-self.lambda_rate * self.finished_epochs)
        elif self.lr_decay_method == "inverse":
            self.learning_rate = self.initial_lr / (1 + self.lambda_rate * self.finished_epochs)
        else:
            raise ValueError("Not a valid decay method.")

    def early_stopping(self):
        """For the moment just simple early stopping but in the future do various methods.

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
        accuracy, loss, _, _ = self.testing.exam()
        self.accuracies.append(accuracy)
        self.losses.append(loss)

        # add info about validation set
        if self.validation_part > 0:
            accuracy_validation, loss_validation, _, _ = self.testing.exam(
                self.validation_images, self.validation_values
            )
            self.validation_exams.append(accuracy_validation)
            self.validation_losses.append(loss_validation)

        # update learning rate
        self.learning_rate_update()

    def learning_rate_update(self):
        if self.lr_decay_method != "":
            self.lr_decay()

    def SGD(self, epoch: int = 1, batch_size: int = 1, to_save: str = ""):
        """Training the model wiht or without batches (if no batches batch_size = 1).

        Args:
            epoch (int, optional): number of iterations through the dataset. Defaults to 5.
            batch_size (int): size of the batch
            to_save (str): if given saves the model to the file every 10 iterations
        """
        num_batches = self.training_images.shape[0] // batch_size

        for e in trange(epoch, desc="Epochs"):

            # permutate the data for new batches
            perm = np.random.permutation(self.training_images.shape[0])

            self.training_images = self.training_images[perm]
            self.training_values = self.training_values[perm]

            self.training_iteration(batch_size, num_batches)

            self.end_iteration()

            if to_save != "" and e % 10 == 0:
                save_model(self.model, self, to_save, checkpoint=True, minus_y=True)

            # early stopping
            if self.early_stop and e >= self.min_epoch:
                b = self.early_stopping()
                if b:
                    self.count_stop += 1
                else:
                    self.count_stop = 0

                if self.count_stop >= self.patience:
                    if to_save != "":
                        save_model(self.model, self, to_save, checkpoint=True, minus_y=True)
                    print(f"Early stop at epoch {e}")
                    break
