import tkinter as tk
from PIL import Image, ImageDraw, ImageColor
from cnn.model import Model
import numpy as np

class Draw:
    def __init__(self, model:Model, size_canva:int=400, bg:str="black", bool_color:bool=False, width:int=20, color_fill:str="white"):
        
        self.root = tk.Tk()
        self.root.title("Draw")

        self.canvas = tk.Canvas(self.root, width=size_canva, height=size_canva, bg=bg)
        self.canvas.pack()

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

        self.create_buttons()

        self.root.mainloop()


    def create_buttons(self):
        prediction = tk.Button(self.root, text="Predict", command=self.predict)
        prediction.pack(side=tk.LEFT)

        close = tk.Button(self.root, text="Close", command=self.root.destroy)
        close.pack(side=tk.LEFT)

        cleaning = tk.Button(self.root, text="Clean", command=self.clean)
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

        image.show()

        pixels = np.asarray(image).reshape(1, h, w, c)

        out = self.model.forward(pixels)

        print(f"I think this is a {self.model.labels[self.model.choice(out).item()]}")
