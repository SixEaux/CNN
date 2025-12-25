import numpy as np


class Convolutional:
    pass



def convolution_forward(self, image, kernel):
    return 

def convolution_backward_input(self, image, kernel):
    return

def convolution_backward_weights(self, image, kernel):
    return

def backconvolution(self, activation, dapres, filtre): # FAIRE LA BACKPROP DE CONVOLUTION
    #pad image pour delta
    #convolution comme avant mais en inversant kernel

    gradc = self.convolution(activation, dapres)

    newdelta = self.convolution(dapres, np.flip(filtre, axis=(2,3)), mode="full", reverse=True)

    return gradc, newdelta