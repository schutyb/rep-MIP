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
rgbscale = tifffile.imread('/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Data-Model/rgb.ome.tiff')

fig, ax = plt.subplots(1, 2, figsize=(16, 10))
ax[0].imshow(imhe)
ax[0].axis('off')
ax[1].imshow(imcolor)
ax[1].axis('off')


fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
ax2.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
phlib.circle_lines(ax2, phases)
plt.show()
