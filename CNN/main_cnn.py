from Import_data import takeinputs
from Parameters import Parametros
from CNN_numpy import CNN
import time

base = "mnist"
inputs = takeinputs(base) #"mnist" #"fashion" #ciphar-10

val, pix, qcmval, qcmpix, labels = inputs

convlay = []

lay = [(100, "sigmoid"), (50, "sigmoid"), (10, "softmax")]

parametros = Parametros(pix=pix, vales=val, qcmpix=qcmpix, qcmval=qcmval, labels=labels,
                        infolay=lay, infoconvlay=convlay, iterations=10, coefcv=0.18, base=base, 
                        batch=1)

g = CNN(parametros)

# g.train()
#
# printgray(g.pix[10])
#
# g.tauxerreur()

# MODEL ENTRAINÉ

# g.importmodel("BestModels/bestmodelmnist")
#
# t0 = g.tauxerreur()
#
# print(g.base)
#
# for i in range(10):
#     g.TryToDraw()
#
# t = g.tauxerreur()
#
# if t >= t0:
#     print("ME HE SUPERADO MUCHO!!!!")
#     g.exportmodel("BestModels/bestmodelmnist")


# MODELE A ENTRAINÉ

print("je commence a mentrainer")
t = time.time()

g.train()

print("jai fini en :", time.time()-t)
g.tauxerreur()

