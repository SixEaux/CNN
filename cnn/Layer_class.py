class Layer:
    def __init__(self):
        pass
    
    def update_weight(self, dw, db, mr, batch, method="momentum"):

        if method == "momentum":
            self.mk = mr*self.mk + dw
        else:
            raise ValueError("Not a known method ")

        self.weight -= self.learning_rate * (dw + self.mk) / batch

        self.bias -= self.learning_rate * db / batch