import numpy as np

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
        save_errors: bool = False,
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
        
        out, loss = self.model.forward(images_test, values_test, test=True)
        preds = self.model.choice(out)
        accuracy = np.mean(preds == values_test) * 100

        if save_errors:
            errors = {i: None for i in range(len(labels))}

            for num in errors.keys():
                y_true = values_test.ravel()
                y_pred = preds.ravel()
                values_num = (y_true == num) & (y_pred != num)

                indexes = np.nonzero(values_num)

                errors[num] = images_test[indexes]
        else:
            errors = None

        return accuracy, np.mean(loss), preds, errors
