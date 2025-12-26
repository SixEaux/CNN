import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt

def converttogreyscale(rgbimage):
        return np.tensordot(rgbimage,np.array([0.299, 0.587, 0.114]), (1, 2))

def paddington(image, padavant, padapres): #padavant ce qu'on ajoute a la ligne et l'autre est evident
        return np.pad(image, ((0,0), (padavant, padapres), (padavant, padapres))) # padding

def choix(self, y):
        return np.argmax(y,axis=0)

def vecteur(self, val):
        if self.lenbatch == 1:
            newval = [val]
        else:
            newval = val
        return np.eye(10)[newval].T


def printbasesimple(base):
        print(tabulate(base.reshape((28, 28))))

def printgray(base, titre="", dims=(28, 28)):
    img = base.reshape(dims)
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title(titre)
    plt.colorbar(label='Value')
    plt.show()

def printimage(base, titre=""):
    img = base.transpose(1,2,0)
    plt.imshow(img)
    plt.axis("off")
    plt.title(titre)
    plt.show()