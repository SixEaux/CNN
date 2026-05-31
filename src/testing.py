import numpy as np
from tqdm import trange

from src.import_data import import_data
from src.model import Model


class Testing:
    def __init__(self, dataset: str, model: Model, loaded_data: tuple = None):
        """Test the accuracy of the model.

        Args:
            dataset (str): dataset used
            model (Model): model to test
        """
        data = import_data(dataset) if loaded_data is None else loaded_data  # import data needed
        self.testing_images = data.test_images
        self.testing_values = data.test_values
        self.labels = data.labels
        self.model = model

    def exam(
        self,
        images_test: np.ndarray = None,
        values_test: np.ndarray = None,
        labels: np.ndarray = None,
        batch_size: int = None,
    ):
        """Test accuracy of the model.

        Args:
            images_test (np.ndarray, optional): images to test on. Defaults to None.
            values_test (np.ndarray, optional): values to test on. Defaults to None.

        Returns:
            tuple: accuracy of the model and mean loss
        """

        if images_test is None and values_test is None and labels is None:
            images_test = self.testing_images
            values_test = self.testing_values
            labels = self.labels
        
        if batch_size is None:
            batch_size = self.testing_images.shape[0]
        
        num_batches = self.testing_images.shape[0] // batch_size
        accuracies = []
        losses = []

        for batch in trange(num_batches, desc="Testing"):
            batch_index = np.arange(batch * batch_size, (batch + 1) * batch_size)
            x_batch = self.testing_images[batch_index]
            exp_batch = self.testing_values[batch_index]

            out, loss = self.model.forward(x_batch, exp_batch, test=True)
            preds = self.model.choice(out)
            accuracy = np.mean(preds == exp_batch) * 100
            accuracies.append(accuracy)
            losses.append(loss)

        return np.mean(accuracies), np.mean(losses), preds
