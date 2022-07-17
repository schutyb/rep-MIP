"""
Here is made an study of melanocytes, given three regions of a melanoma where we know there are melanocytes
we get the HSI images and do:
1 - Do phasor analysis; with g and s plot the phasor and obtain the colored image.
2 - There is a channel image to compared with the colored image.
3 - Plot also the segmented H&E image
4 - Plot the spectral intensity graph of some ROI's
"""

import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as phlib
import numpy as np
from skimage.filters import median, gaussian
from matplotlib import colors


path = '/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Melanocytes/mela_16952/'
file_lsm = '16952_SP_R1.lsm'
filename = '16952_phasor_R1.ome.tiff'
channel = '16952_channel_R1.lsm'

file_generator = False
if file_generator:
    im = tifffile.imread(path + file_lsm)
    dc, g, s, _, _ = phlib.phasor(im)
    dc = gaussian(dc, sigma=1.2)
    cont = 0
    while cont < 2:
        g = median(g)
        s = median(s)
        cont = cont + 1
    # umbralización
    dc = np.where(dc > 1, dc, np.zeros(dc.shape))
    g = np.where(dc > 1, g, np.zeros(g.shape))
    s = np.where(dc > 1, s, np.zeros(s.shape))
    ph = np.round(np.angle(g + s * 1j, deg=True))
    md = np.sqrt(g ** 2 + s ** 2)
    # guardo los datos en un ome.tiff
    data = phlib.generate_file(path + filename, [dc, g, s, md, ph])

im = tifffile.imread(path + filename)
ph = im[4]
x = np.concatenate(im[1])
y = np.concatenate(im[2])
# para encontrar el intervalo de ph tomo ph y hago el histograma de 0 a 360 bins y ahi corto los extremos
hist = np.histogram(np.concatenate(ph), np.arange(360))
hist[0][0] = 0
bins = np.arange(0, 359)
seg = phlib.segment_thresholding(hist[0], bins, 0.01)
rgb = phlib.colored_image(ph, np.asarray([min(seg), max(seg)]))
im_channel = tifffile.imread(path + channel)[1]
im_he = plt.imread('/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Melanocytes/mela_16952/16952_HE.png')


fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0, 0].imshow(im_channel, cmap='gray', interpolation='spline16')
ax[0, 0].axis('off')
ax[0, 1].imshow(im_he, interpolation='none')
ax[0, 1].axis('off')
ax[0, 1].set_title('H&E stain')
ax[1, 0].hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-0.75, 0.75], [0, 1]])
ax[1, 0].set_title('Phasor')
phlib.phasor_circle(ax[1, 0])
ax[1, 1].imshow(rgb, interpolation='spline16')
ax[1, 1].axis('off')
ax[1, 1].set_title('Fluorescent')
plt.show()
