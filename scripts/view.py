import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as phlib
import numpy as np


plot = False
if plot:
    path = '/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/Breslow/Phasor/'
    filename = '15579.ome.tiff'
    im = tifffile.imread(path + filename)

    plt.figure(1)
    plt.imshow(im[0], cmap='gray')
    plt.show()


im = tifffile.imread('/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Melanocytes/img del modelo/15477_SP_r2.lsm')
im = phlib.phasor(im)

'''
hist = np.histogram(np.concatenate(phase, bins=np.array(360)))
rgb = phlib.colored_image(phase, phinterval=)
plt.figure(1)
plt.imshow(rgb)
plt.show()
'''
