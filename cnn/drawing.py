import tkinter as tk
from PIL import Image, ImageDraw, ImageColor
import numpy as np

from cnn.model import Model
from cnn.training import Training

from cnn.save_model import save_model

class Draw:
    def __init__(self, model:Model, train:Training=None, learn_bool:bool=False, size_canva:int=400, bg:str="black", bool_color:bool=False, width:int=20, color_fill:str="white"):
        
        self.root = tk.Tk()
        self.root.title("Draw")

        self.canva_frame = tk.Frame(self.root)
        self.other_frame = tk.Frame(self.root)

        self.canva_frame.pack()
        self.other_frame.pack()

        self.canvas = tk.Canvas(self.canva_frame, width=size_canva, height=size_canva, bg=bg)
        self.canvas.pack()
        self.text_input = tk.Text(self.other_frame, height=1, width=20, )
        self.text_input.pack()

        self.rgb_bg = ImageColor.getcolor(bg, "L")

        self.image = Image.new("L" if not bool_color else "RGB", (size_canva, size_canva), self.rgb_bg)
        self.drawing = ImageDraw.Draw(self.image)

        self.width = width
        self.color_fill = color_fill
        self.bool_color = bool_color
        self.size_canva = size_canva

        self.pos_x, self.pos_y = None, None

        self.bool_drawing = False

        self.model = model
        self.labels_reversed = {self.model.labels[k]:k for k in self.model.labels.keys()}

        self.train = train
        self.learn_bool = learn_bool

        self.create_buttons()

        self.root.mainloop()


    def create_buttons(self):
        prediction = tk.Button(self.other_frame, text="Predict", command=self.predict)
        prediction.pack(side=tk.LEFT)

        close = tk.Button(self.other_frame, text="Close", command=self.close)
        close.pack(side=tk.LEFT)

        cleaning = tk.Button(self.other_frame, text="Clean", command=self.clean)
        cleaning.pack(side=tk.LEFT)

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)


    def start_drawing(self, event):
        self.bool_drawing = True
        self.pos_x, self.pos_y = event.x, event.y
    
    def stop_drawing(self, event):
        self.bool_drawing = False
    
    def draw(self, event):
        if self.bool_drawing:
            x, y = event.x, event.y

            self.canvas.create_line((self.pos_x, self.pos_y, x, y), fill=self.color_fill, width=self.width)

            self.drawing.line([self.pos_x, self.pos_y, x, y], fill=self.color_fill, width=self.width)

            self.pos_x, self.pos_y = x, y

    def clean(self):
        self.image = Image.new("L" if not self.bool_color else "RGB", (self.size_canva, self.size_canva), self.rgb_bg)
        self.drawing = ImageDraw.Draw(self.image)

        self.canvas.delete("all")

    def predict(self):
        image = self.image.copy()

        h, w, c = self.model.input_size

        image = image.resize((w, h), Image.Resampling.LANCZOS)

        pixels = np.asarray(image).reshape(1, h, w, c)

        if self.learn_bool and self.train is not None:
            real_value = self.text_input.get("1.0").strip()

            if real_value:
                try:
                    value = int(real_value)
                except:
                    value = self.labels_reversed[real_value]
            else:
                value = None
        
            out = self.model.forward(pixels, np.array(value) if value else None)

            pred = self.model.choice(out)[0,0]

            print(f"I think this is a {self.model.labels[pred]}")

            if value is not None:
                self.model.backward(1, self.train.learning_rate, 0)

            if pred != value:
                print(f"I was actually wrong and I am going to train on it.")
        
        else:
            out = self.model.forward(pixels)

            pred = self.model.choice(out).item()

            print(f"I think this is a {self.model.labels[pred]}")

    def close(self):
        self.root.destroy()

        if self.train is not None:
            while True:
                save = input("Do you want to save the model? (y/n)")

                if save == "y":
                    file = input("File name: ")
                    save_model(self.model, self.train, file)
                elif save == "n":
                    break
                else:
                    print("Invalid input.")
