class Flattening:
    def __init__(self):
        pass

    def forward(self, x):
        return x.reshape((-1,1))
    
    def backward(self):
        pass