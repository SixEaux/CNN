from Import_data2 import import_data
from tqdm import trange


class Training:
    def __init__(self, dataset:str, model):
        self.training_images, self.training_values, _, _ = import_data(dataset)
        self.model = model
    
    def training_simple(self, epoch=1):
        for e in trange(epoch, desc="Epochs"):
            for p in trange(self.training_images.shape[0], desc="Images"):
                image = self.training_images[p].reshape(-1, 1)
                loss = self.model.forward(image, self.training_values[p])
                self.model.backward()
