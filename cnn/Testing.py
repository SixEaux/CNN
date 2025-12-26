import numpy as np

from cnn.Import_data2 import import_data
from cnn.Model import Model


class Testing:
    def __init__(self, dataset: str, model: Model):
        """Test the accuracy of the model.

        Args:
            dataset (str): dataset used
            model (Model): model to test
        """
        _, _, self.testing_images, self.testing_values = import_data(
            dataset)  # import data needed
        self.model = model

    def exam(self):
        """Test accuracy of the model.

        Returns:
            float: accuracy of the model
        """
        preds = self.model.prediction(
            self.testing_images.reshape(self.testing_images.shape[0], -1))
        accuracy = np.mean(preds == self.testing_values) * 100
        return accuracy
