from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import os

from cnn.import_data import import_data
from cnn.model import Model
from cnn.testing import Testing

from cnn.save_model import save_model



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
        """

    def __init__(self, dataset: str, model: Model, testing: Testing, learning_rate:float, normalize: str = "division",
                 lr_decay:str="",lambda_rate:float=0, momentum_rate:float=0, validation_part:float=0, early_stop:bool=False):
        self.dataset = dataset
        self.validation_part = validation_part
        self.training_images, self.training_values, self.validation_images, self.validation_values, _, _ = import_data(self.dataset, validation_part)  # import data needed
        self.model = model
        self.testing = testing
        self.learning_rate = learning_rate

        self.initial_lr = learning_rate
        self.lr_decay_method = lr_decay
        self.lambda_rate = lambda_rate
        self.momentum_rate = momentum_rate

        self.finished_epochs = 0 # number of finished epochs
        self.learning_rates = [self.learning_rate] # keep track of the learning rates of each iteration
        self.losses = []  # keep track of the losses of each iteration
        self.accuracies = []  # keep track of the accuracy of each iteration
        self.validation_exams = [] # keep track of the accuracy of each iteration on validation set
        self.validation_losses = [2, 2] # keep track loss in validation sample of this iteration

        self.early_stop = early_stop

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

    def training_iteration(self, batch_size:int, num_batches:int):
        """Do an iteration of training.

        Args:
            batch_size (int): size of the batch
            num_batches (int): number of batches that divides dataset
        """
        for batch in trange(num_batches, desc="Batch"):
            x_batch = self.training_images[batch *
                                            batch_size:(batch+1)*batch_size]
            exp_batch = self.training_values[batch *
                                                batch_size:(batch+1)*batch_size]

            self.model.forward(x_batch, exp_batch)

            self.model.backward(batch_size, self.learning_rate)

    def lr_decay(self):
        """Decay the learning rate based on the methof used.

        Raises:
            ValueError: if the method of learning decay is unkown
        """
        if self.lr_decay_method == "":
            return 
        elif self.lr_decay_method == "exponential":
            self.learning_rate = self.initial_lr * np.exp(-self.lambda_rate*self.finished_epochs)
        elif self.lr_decay_method == "inverse":
            self.learning_rate = self.initial_lr / (1 + self.lambda_rate * self.finished_epochs)
        else:
            raise ValueError("Not a valid decay method.")
    
    def early_stopping(self):
        """Decay learning rate. For the moment just simple early stopping but in the future do various methods.

        Returns:
            bool: if True stop
        """
        if self.validation_losses[-2] < self.validation_losses[-1]: #if the loss from previous iteration is smaller than this one
            return True
        else:
            return False

    def end_iteration(self):
        """Stuff to do after the iteration.
        """
        self.finished_epochs += 1
        self.lr_decay()

        # add important info of this iteration
        accuracy, loss = self.testing.exam()
        self.accuracies.append(accuracy)
        self.losses.append(loss)
        self.learning_rates.append(self.learning_rate)

        # add info about validation set
        if self.validation_part > 0:
            accuracy_validation, loss_validation = self.testing.exam(self.validation_images, self.validation_values)
            self.validation_exams.append(accuracy_validation)
            self.validation_losses.append(loss_validation)

    def SGD(self, epoch:int=1, batch_size:int=1, to_save:str=""):
            """Training the model wiht or without batches (if no batches batch_size = 1).

            Args:   
                epoch (int, optional): number of iterations through the dataset. Defaults to 5.
                batch_size (int): size of the batch
                to_save (str): if given saves the model to the file every 10 iterations
            """
            num_batches = (self.training_images.shape[0] // batch_size)
            for e in trange(epoch, desc="Epochs"):
                self.training_iteration(batch_size, num_batches)

                self.end_iteration()

                if to_save != "" and e % 10 == 0:
                    save_model(self.model, self, to_save, checkpoint=True)

                #early stopping
                if self.early_stop and self.early_stopping():
                    break

    def plot_smthg(self, smthg:np.ndarray, save_to:str="", title:str="", x_title:str="", y_title:str="", show:bool=False):
        """Plot stuff from model.

        Args:
            smthg (np.ndarray): stuff to plot
            save_to (str): folder where to save the plot
            title (str, optional): title of the plot and saved under this name. Defaults to "".
            x_title (str, optional): label of x. Defaults to "".
            y_title (str, optional): label of y. Defaults to "".
            show (bool, optional): if true show the plot. Defaults to False.
        """

        plt.figure(figsize=(10, 6))
        plt.plot(smthg)
        plt.title(title)
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Loss")
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        if save_to != "":
            dir_plots = os.path.join("outputs", "plots")
            new_folder_path = os.path.join(dir_plots, save_to)
            os.makedirs(new_folder_path, exist_ok=True)
            plt.savefig(os.path.join(new_folder_path, title))

        if show:
            plt.show()