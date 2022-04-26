import tifffile
import PhasorLibrary as Ph
import numpy as np


froute = str('/home/bruno/Documentos/TESIS/TESIS/Experimentos/exp_bordes/img_1x1/lsm/')
fname = str('exp_1x1_nevo_2.lsm')
f = froute + fname

f = str('/home/bruno/Descargas/Image 4.lsm')
im = tifffile.imread(f)

g, s, _, _, dc = Ph.phasor(im, harmonic=1)
dc = np.mean(im, axis=0)
for i in range(5):
    Ph.interactive(dc, g, s, 0.1)


"""g = np.zeros([im.shape[0], im.shape[2], im.shape[3]])
s = np.zeros([im.shape[0], im.shape[2], im.shape[3]])
dc = np.zeros([im.shape[0], im.shape[2], im.shape[3]])

for i in range(im.shape[0]):
    g[i], s[i], _, _, dc[i] = Ph.phasor(im[i], harmonic=1)
"""


