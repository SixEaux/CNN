from src.load_model import load_model
from src.visualize import visual_image, visual_outputs, confusion_matrix_plot, plot_smthg

file = "fashion_optimised"

model, trainer, test = load_model(file)

t = test.exam(save_errors=True)

# confusion_matrix_plot(t[2], test.testing_values, test.labels.values())

visual_image(t[3][6][:40])

