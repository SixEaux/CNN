import numpy as np

class Pooling:
    pass



def pooling(self, image):
        division = np.lib.stride_tricks.sliding_window_view(image, (self.lenkernelpool, self.lenkernelpool), axis=(1, 2))[:, ::self.lenkernelpool, ::self.lenkernelpool]
        return np.average(division, axis=(3, 4))


def backpool(self, dapres, dimsortie): # REVENIR AUX MEMES DIMENSIONS QU'AVANT POOLING
        moyenne = dapres / (self.lenkernelpool * self.lenkernelpool)

        if dimsortie[1] % self.lenkernelpool == 0: #si pile
            output = np.zeros(dimsortie)

            for d in range(dapres.shape[0]):
                output[d] = np.repeat(np.repeat(moyenne[d], self.lenkernelpool, axis=0), self.lenkernelpool, axis=1) #on recree un kernel avec les dimensions

        else:
            c, h, l = dimsortie

            dif = h % self.lenkernelpool, l % self.lenkernelpool #si pas pile

            newh, newl = h - (dif[0]), l - (dif[1])

            output = np.zeros(dimsortie)

            for d in range(dapres.shape[0]):
                output[d, :newh, :newl] = np.repeat(np.repeat(moyenne[d], self.lenkernelpool, axis=0), self.lenkernelpool, axis=1)

        return output
