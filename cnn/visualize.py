import numpy as np
from matplotlib import pyplot as plt
from cnn.save_model import extract_model_state

from cnn.training import Training
from cnn.testing import Testing
from cnn.model import Model


def visual_image(image:np.ndarray):
    """Print image.

    Args:
        image (np.ndarray): DIM = (h, w, channels)
    """
    
    if image.shape[-1] == 1:
        image = image.squeeze()

    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    
    plt.show()


def visual_outputs(model:Model, index_layer:int, cols:int=8):
    output = model.saved_outputs[index_layer] # either (1, h, w, c) or (1, -1)

    if output.ndim == 4:
        b, h, w, c = output.shape
        assert b == 1, "Visualization only with one image at a time."

        rows = int(np.ceil(c / cols))

        fig, ax = plt.subplots(rows, cols, figsize=(cols, rows))
        ax = np.atleast_2d(ax)

        for i in range(rows * cols):
            row, col = i // cols, i % cols
            ax[row, col].axis("off")

            if i < c:
                out = output[0, :, :, i]
                ax[row, col].imshow(out, cmap="gray")

        plt.show()
    
    elif output.ndim == 2:
        plt.figure(figsize=(8, 2))
        plt.imshow(output, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.show()
    
    else:
        raise ValueError("What?")