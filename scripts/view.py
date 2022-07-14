import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as phlib
import numpy as np
from skimage.filters import median, gaussian

file_generator = False
if file_generator:
    im = tifffile.imread('/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Melanocytes/img del '
                         'modelo/15477_SP_16avg_1A.lsm')
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
    filename = '/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Melanocytes/img del ' \
               'modelo/phasor/15477_SP_16avg_1A.ome.tiff '
    data = phlib.generate_file(filename, [dc, g, s, md, ph])

file = '/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Melanocytes/img del modelo/phasor/15477_SP_r2.ome.tiff'
im = tifffile.imread(file)
md = im[3]
ph = im[4]
# para encontrar el intervalo de ph tomo ph y hago el histograma de 0 a 360 bins y ahi corto los extremos
hist = np.histogram(np.concatenate(ph), np.arange(360))
hist[0][0] = 0
bins = np.arange(0, 359)
seg = phlib.segment_thresholding(hist[0], bins, 0.01)
rgb = phlib.colored_image(ph, np.asarray([min(seg), max(seg)]))

channel = '/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Melanocytes/img del modelo/15477_channel_r2.lsm'
im_channel = tifffile.imread(channel)[1]


plt.figure(1)
plt.imshow(rgb, interpolation='spline16')

plt.figure(2)
plt.imshow(im_channel, cmap='gray', interpolation='spline16')
plt.show()
