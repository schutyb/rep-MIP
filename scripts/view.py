import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as phlib
import numpy as np
from matplotlib import colors


plot = False
if plot:
    path = '/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/HSI/'
    filename = '15719_SP_Tile_04x04.lsm'
    im = tifffile.imread(path + filename)

    plt.figure(1)
    plt.imshow(im[0], cmap='gray')
    plt.show()

#  ploteo HE, pseudocolor, phasor
im = tifffile.imread('/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/Phasor/16256.ome.tiff')
x = np.concatenate(im[1])
y = np.concatenate(im[2])
imcolor = plt.imread('/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/Fluorescent/png/18852.png')
imhe = plt.imread('/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/H&E/png/18852b.png')
phases = np.asarray([70, 85, 100, 120, 135])

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(imhe)
ax[0].axis('off')
ax[1].hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
phlib.circle_lines(ax[1], phases)
ax[2].imshow(imcolor)
ax[2].axis('off')

plt.show()
