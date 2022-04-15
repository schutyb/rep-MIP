import numpy as np
from PhasorLibrary import phasor, phasor_plot
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import median, gaussian
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


'''
En este codigo utilizo imagenes con diferentes promedios para comparar
calculo los g y s con 16 promedios y los tomo como gold standr y luego calculo con los demas promedios 
'''

#  The 16 averages is the gold standar
file = str('/home/bruno/Documentos/TESIS/TESIS/Experimentos/exp_avg/test_16_mean.lsm')
gs = tifffile.imread(file)
g, s, _, _, dc = phasor(gs)

g1 = [g]
s1 = [s]
dc1 = [dc]
ity = [3]

plt.figure(1)
plt.imshow(dc)
plt.show()
plt.show()

fig, _, _ = phasor_plot(g1, s1, dc1, ity)
plt.show()
