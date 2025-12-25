from Import_data2 import import_data
import numpy as np
from tqdm import trange

class Testing:
    def __init__(self, dataset, model):
        _, _, self.testing_images, self.testing_values = import_data(dataset)
        self.model = model
    
    def exam(self):
        well_predicted = 0
        for p in trange(self.testing_images.shape[0]):
            image = self.testing_images[p].reshape(-1, 1)
            pred = self.model.prediction(image, self.testing_values[p])
            well_predicted += pred
        return well_predicted * 100 / self.testing_images.shape[0]