import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

from cnn.training import Training
from cnn.testing import Testing
from cnn.model import Model
from cnn.helpers import conversation_save

def plot_smthg(smthg:np.ndarray, save_to:str="", title:str="", x_title:str="", y_title:str="", show:bool=False, minus_y:bool=False):
        """Plot stuff from model.

        Args:
            smthg (np.ndarray): stuff to plot
            save_to (str): folder where to save the plot
            title (str, optional): title of the plot and saved under this name. Defaults to "".
            x_title (str, optional): label of x. Defaults to "".
            y_title (str, optional): label of y. Defaults to "".
            show (bool, optional): if true show the plot. Defaults to False.
            minus_y (bool): directly save without asking
        """

        plt.figure(figsize=(10, 6))
        plt.plot(smthg)
        plt.title(title)
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Loss")
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        if show:
            plt.show()

        

        def save():
            dir_plots = os.path.join("outputs", "plots")
            new_folder_path = os.path.join(dir_plots, save_to)
            os.makedirs(new_folder_path, exist_ok=True)
            plt.savefig(os.path.join(new_folder_path, title))
            plt.close()

        conversation_save(save, save_to, minus_y)



def visual_image_meh(image:np.ndarray, cols:int=8):
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

def visual_outputs_meh(model:Model, index_layer:int, cols:int=8):
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

def confusion_matrix_plot(prediction:np.ndarray, expected:np.ndarray, labels:np.ndarray):
        if prediction.ndim != 1:
            prediction = prediction.reshape(-1,)
        
        if expected.ndim != 1:
            expected = expected.reshape(-1,)
        

        confusion = confusion_matrix(expected, prediction)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)

        disp.plot()

        plt.show()
    

def visual_image(
    image: np.ndarray,
    cols: int = 8,
    cmap: str = "gray",
    normalize: bool = True,
    title: str = None, 
    save_to:str="",
    minus_y:bool=False,
    show:bool=True):

    def _prepare(img):
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        if normalize and img.ndim == 2:
            vmin, vmax = np.percentile(img, (1, 99))
            img = np.clip(img, vmin, vmax)
        return img

    if image.ndim == 3:
        img = _prepare(image)

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap=cmap if img.ndim == 2 else None)
        plt.axis("off")
        if title:
            plt.title(title, fontsize=14)
        plt.tight_layout(pad=0)
        plt.show()
        return

    if image.ndim != 4:
        raise ValueError("Image must be 3D or 4D array.")

    n = image.shape[0]
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 1.5, rows * 1.5),
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            img = _prepare(image[i])
            ax.imshow(img, cmap=cmap if img.ndim == 2 else None)

    if title:
        fig.suptitle(title, fontsize=16)

    if show:
        plt.show()

    def save():
        dir_plots = os.path.join("outputs", "images")
        new_folder_path = os.path.join(dir_plots, save_to)
        os.makedirs(new_folder_path, exist_ok=True)
        plt.savefig(os.path.join(new_folder_path, title))
        plt.close()

    conversation_save(save, save_to, minus_y, type_thing="images")





def visual_outputs(
    model,
    index_layer: int,
    cols: int = 8,
    cmap: str = "viridis",
    normalize: bool = True,
    save_to:str="",
    minus_y:bool=False
):

    output = model.saved_outputs[index_layer]

    if output.ndim == 4:
        b, h, w, c = output.shape
        if b != 1:
            raise ValueError("Visualization supports batch size = 1 only.")

        rows = int(np.ceil(c / cols))
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 1.4, rows * 1.4),
            constrained_layout=True
        )
        axes = np.atleast_2d(axes)

        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < c:
                fmap = output[0, :, :, i]
                if normalize:
                    vmin, vmax = np.percentile(fmap, (1, 99))
                    fmap = np.clip(fmap, vmin, vmax)
                ax.imshow(fmap, cmap=cmap)

        title = f"Layer {index_layer} – Feature maps"
        fig.suptitle(title, fontsize=16)
        plt.show()
        

    if output.ndim == 2:
        plt.figure(figsize=(10, 2))
        plt.imshow(output, aspect="auto", cmap=cmap)
        plt.colorbar(fraction=0.02, pad=0.02)
        title = "Layer {index_layer} – Vector output"
        plt.title(title, fontsize=14)
        plt.yticks([])
        plt.tight_layout()
        plt.show()
    


    def save():
        dir_plots = os.path.join("outputs", "images")
        new_folder_path = os.path.join(dir_plots, save_to)
        os.makedirs(new_folder_path, exist_ok=True)
        plt.savefig(os.path.join(new_folder_path, title))
        plt.close()

    conversation_save(save, save_to, minus_y, type_thing="images")



