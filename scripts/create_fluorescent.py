import PhasorLibrary as phlib
import tifffile
from skimage.filters import median, gaussian
import numpy as np
import os
import matplotlib.pyplot as plt


ometiff = True
if ometiff:
    path = '/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/Phasor/algo/'
    files_names = os.listdir(path)
    for k in range(len(files_names)):
        im = tifffile.imread(path + files_names[k])
        fname = files_names[k][:5]
        # Obtencion del phinterval
        hist = np.histogram(np.concatenate(im[4]), np.arange(0, 180, 1))
        hist[0][0] = 0
        phinterval = phlib.segment_thresholding(hist[0], np.arange(0, 179, 1), 0.001)
        rgb = phlib.colored_image(im[4], phinterval)
        f = '/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/Fluorescent/' + fname + '.ome.tiff'
        phlib.generate_file(f, rgb)

png = True
path = '/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/Fluorescent/'
files_names = os.listdir(path)
for k in range(len(files_names)):
    im = tifffile.imread(path + files_names[k])
    fname = files_names[k][:5]
    f = path + fname + '.png'
    plt.imsave(f, im)

