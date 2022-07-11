import numpy as np
import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as PhLib
import csv
from matplotlib import colors


# imagen a color rgb que tiene la LUT del modelo
path = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/Modelo_melanomas/'
with open(path + 'model_parameters.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        (', '.join(row))

# obtengo los parameters del modelo
phmin = float(row[0]) - 3
phmax = float(row[1]) + 3
mdmin = float(row[2])
mdmax = float(row[3])
phinterval = np.asarray([phmin, phmax])
mdinterval = np.asarray([mdmin, mdmax])

name = '16952'
path2 = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/'
im = tifffile.imread(path2 + name + '.ome.tiff')
imcolor_umb, _ = PhLib.color_normalization(im[3], im[4], phinterval, mdinterval)
imcolor, _ = PhLib.color_normalization(im[3], im[4], phinterval, mdinterval, modulation=False)
rgb = tifffile.imread(path + 'rgb.ome.tiff')
x, y = PhLib.phasor_threshold(im[1], im[2], im[3], im[4], phinterval, mdinterval)

plot = False
if plot:  # phasor plot and pseudocolor rgb
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [0, 1]])
    ax[0].set_title('Phasor')
    PhLib.phasor_circle(ax[0])
    ax[1].imshow(imcolor, interpolation='none')
    ax[1].set_title('Colored image')
    ax[1].axis('off')
    plt.show()

plot_3 = False
if plot_3:  # ploteo las tres cosas, phasor, espectro y rgb
    spectrum = tifffile.imread(path + 'rgb.ome.tiff')
    xlabels = [0.332, 0.368, 0.404, 0.44, 0.476, 0.512, 0.548, 0.584, 0.62, 0.656, 0.69]

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [0, 1]])
    ax[0].set_title('Phasor')
    PhLib.phasor_circle(ax[0])
    ax[0].axis('off')
    ax[0].set_ylim(0, 1)

    ax[1].imshow(rgb, interpolation='none', extent=[0, 180, phinterval[0], phinterval[1]])
    ax[1].set_aspect(2)
    ax[1].set_xlabel('Modulation')
    ax[1].set_ylabel('Phase')
    ax[1].set_xticklabels(xlabels)
    ax[1].set_title('Spectrum')

    ax[2].imshow(imcolor, interpolation='none')
    ax[2].set_title('Colored image')
    ax[2].axis('off')
    plt.show()

plt_both = False
if plt_both:  # ploteo los dos phasors, con umbralización y sin
    g = np.concatenate(im[1])
    s = np.concatenate(im[2])
    fig2, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [0, 1]])
    ax[0].set_title('Phasor')
    PhLib.phasor_circle(ax[0])
    ax[1].hist2d(g, s, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [0, 1]])
    ax[1].set_title('Phasor')
    PhLib.phasor_circle(ax[1])
    plt.show()
