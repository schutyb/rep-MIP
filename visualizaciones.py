import PhasorLibrary as Ph
import numpy as np
import PhasorPy
import tifffile


# f1 = str('/home/bruno/Documentos/TESIS/TESIS/IMAGENES LSM/2022/MELANOMAS/16952_SP_Tile_4x3.lsm')
f1 = str('/home/bruno/Documentos/TESIS/TESIS/IMAGENES LSM/2022/MELANOMAS/15477_SP_Tile_6x3.lsm')
# f1 = str('/home/bruno/Documentos/TESIS/TESIS/3x3_2avg.lsm')
im = tifffile.imread(f1)

# Phasor tile
dc1, g1, s1, _, _ = Ph.phasor_tile(im, 1024, 1024)

dc = Ph.concatenate(dc1, 3, 6)
g = Ph.concatenate(g1, 3, 6)
s = Ph.concatenate(s1, 3, 6)

for i in range(2):
    PhasorPy.interactive(dc, g, s, 0.3)


"""g = np.zeros([im.shape[0], im.shape[2], im.shape[3]])
s = np.zeros([im.shape[0], im.shape[2], im.shape[3]])
dc = np.zeros([im.shape[0], im.shape[2], im.shape[3]])

for i in range(im.shape[0]):
    g[i], s[i], _, _, dc[i] = Ph.phasor(im[i], harmonic=1)
"""


