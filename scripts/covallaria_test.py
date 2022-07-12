"""
Here we evaluate the phase/modulation and solo phase color image.
Given the convallaria image witch has many spectral phases, we expect to see a big change in colored images
with and without the modulation information.
0 - The data is a 3x2 HSI, it is used the concatenation and tile phasor algorithms.
1 - Plot the colored images in both cases
2 - Plot the phasor (the is no thresholded phasor, since there is no model, because it is only one image data)
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as PhLib
from skimage.filters import median


path = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/MIP-Data/convallaria/'
# 0 - Phasor analysis: phasor plot and colored image
calcular = False
if calcular:
    im = tifffile.imread(path + 'convallarias_3x2_SP.lsm')  # ROI with border
    dc, g, s, _, _ = PhLib.phasor_tile(im, 1024, 1024)
    dc = PhLib.concatenate(dc, 2, 3, hper=0.06)
    g = PhLib.concatenate(g, 2, 3, hper=0.06)
    s = PhLib.concatenate(s, 2, 3, hper=0.06)

    cont = 0
    while cont < 2:
        g = median(g)
        s = median(s)
        cont = cont + 1

    dc = np.where(dc > 1, dc, np.zeros(dc.shape))
    g = np.where(dc > 1, g, np.zeros(g.shape))
    s = np.where(dc > 1, s, np.zeros(s.shape))
    ph = np.round(np.angle(g + s * 1j, deg=True))
    md = np.sqrt(g ** 2 + s ** 2)

    store = False # guardo los datos en un ome.tiff
    if store:
        filename = path + 'convallaria' + '.ome.tiff'
        data = PhLib.generate_file(filename, [dc, g, s, md, ph])

# 1 - Colored Images
#  Aca tengo que hacer el histograma del módulo y de la fase y sacar los valores del segmento donde umbralizo
#  asi se lo paso a la funcion im_normalization para obtener la pseudocolor
im = tifffile.imread('/home/bruno/Documentos/Proyectos/TESIS/TESIS/MIP-Data/convallaria/convallaria.ome.tiff')
md = im[3]
ph = im[4]

''' bins_ph = np.arange(360)
bins_md = np.linspace(0, 1, 500)
phase = np.concatenate(ph)
modulo = np.concatenate(md)
hist_ph = np.histogram(phase, bins_ph)[0]
hist_ph[0] = 0
hist_md = np.histogram(modulo, bins_md)[0]
hist_md[0] = 0
per = 0.0001
bins_ph = np.arange(359)
bins_md = np.linspace(0, 1, 499)
mdinterval = PhLib.segment_thresholding(hist_md, bins_md, per)
mdinterval = np.asarray([min(mdinterval), max(mdinterval)])
phinterval = PhLib.segment_thresholding(hist_ph, bins_ph, per)
phinterval = np.asarray([min(phinterval), max(phinterval)])
'''

imcolor, _ = PhLib.color_normalization(md, ph, threshold=False)

plt.figure(1)
plt.imshow(imcolor)
plt.axis('off')

PhLib.phasor_plot(np.asarray([im[0]]), np.asarray([im[1]]), np.asarray([im[2]]))
plt.show()
