import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from cnn.training import Training
from cnn.testing import Testing
from cnn.model import Model


def visual_image(image:np.ndarray, cols:int=8):
    """Print image (s). If more than one image num_images is first shape.

    Args:
        image (np.ndarray): DIM = (num_images, h, w, channels) or (h, w, channels)
    """
    
    if image.ndim == 3:
        if image.shape[-1] == 1:
            image = image.squeeze(-1)

        if image.ndim == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
        
        plt.show()
    
    elif image.ndim == 4:
        n, h, w, c = image.shape

        rows = int(np.ceil(n / cols))

        fig, ax = plt.subplots(rows, cols, figsize=(cols, rows))
        ax = np.atleast_2d(ax)

        for i in range(rows * cols):
            row, col = i // cols, i % cols
            ax[row, col].axis("off")

            if i < n:
                im = image[i]
                if im.shape[-1] == 1:
                    im = im.squeeze(-1)

                if im.ndim == 2:
                    ax[row, col].imshow(im, cmap="gray")
                else:
                    ax[row, col].imshow(im)

        plt.show()



def visual_outputs(model:Model, index_layer:int, cols:int=8):
    output = model.saved_outputs[index_layer] # either (1, h, w, c) or (1, -1)

    if output.ndim == 4:
        b, h, w, c = output.shape
        assert b == 1, "Visualization only with one batch at a time."

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


def confusion_matrix_plot(prediction:np.ndarray, expected:np.ndarray):
        if prediction.ndim != 1:
            prediction = prediction.reshape(-1,)
        
        if expected.ndim != 1:
            expected = expected.reshape(-1,)
        

        confusion = confusion_matrix(expected, prediction)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion)

        disp.plot()

        plt.show()