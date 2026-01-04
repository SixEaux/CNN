import numpy as np

from cnn.import_data import import_data
from cnn.model import Model


class Testing:
    def __init__(self, dataset: str, model: Model):
        """Test the accuracy of the model.

        Args:
            dataset (str): dataset used
            model (Model): model to test
        """
        _, _, _, _, self.testing_images, self.testing_values = import_data(dataset)  # import data needed
        self.model = model

    def exam(self, images_test:np.ndarray=None, values_test:np.ndarray=None):
        """Test accuracy of the model.

        Args:
            images_test (np.ndarray, optional): images to test on. Defaults to None.
            values_test (np.ndarray, optional): values to test on. Defaults to None.

        Returns:
            tuple: accuracy of the model and mean loss
        """

        if images_test is None and values_test is None:
            images_test = self.testing_images
            values_test = self.testing_values

        out, loss = self.model.forward(images_test, values_test)
        preds = self.model.choice(out)
        accuracy = np.mean(preds == values_test) * 100

        return accuracy, np.mean(loss)
