import PhasorLibrary as phlib
import tifffile
import numpy as np
import matplotlib.pyplot as plt


im = tifffile.imread('/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/HSI/16952_SP_Tile_04x03.lsm')
dc, g, s, _, _ = phlib.phasor_tile(im, 1024, 1024)
dcc = phlib.concatenate(dc, 3, 4)

phlib.generate_file('/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/HSI/ej/16952_4x4.ome.tiff', dc)
phlib.generate_file('/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/HSI/ej/16952.ome.tiff', dcc)
plt.imsave('/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/HSI/ej/16952.png', dcc)
