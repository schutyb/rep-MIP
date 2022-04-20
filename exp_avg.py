import numpy as np
import PhasorLibrary
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
g, s, _, _, dc = PhasorLibrary.phasor(gs, harmonic=1)


fig = PhasorLibrary.interactive(dc, g, s, 0.2)

