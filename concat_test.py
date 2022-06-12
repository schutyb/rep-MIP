import numpy as np
import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt
from tifffile import imwrite, memmap
import cv2


'''En este codigo voy a probar como llego al phasor de la 2x2, entonces voy a usar una imagen 2x2 y calcular el phasor
de cada canal y luego concatenar g, s, md, ph y dc. Además con esto ya pruebo guardar el ome.tiff comun y el 
comprimido'''


#  Pruebo la funcion concatenar con una imagen sintetica estilo damero.
'''
i1 = cv2.imread('/home/bruno/Documentos/TESIS/gris.png', cv2.IMREAD_GRAYSCALE)
i2 = cv2.imread('/home/bruno/Documentos/TESIS/negro.png', cv2.IMREAD_GRAYSCALE)
dc = np.asarray([i1, i2, i1, i2, i2, i1, i2, i1, i1, i2, i1, i2, i2, i1, i2, i1])
'''

# f1 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Experimentos/caso_18370/lsm/18370_SP_Tile_2x2_b.lsm')
f1 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/base de datos lsm/2022/MELANOMAS/16952_SP_Tile_4x3.lsm')
# f1 = str('/home/bruno/Documentos/TESIS/TESIS/IMAGENES LSM/2022/MELANOMAS/20412_SP_Tile_11x6.lsm')
# f1 = str('/home/bruno/Documentos/TESIS/TESIS/3x3_2avg.lsm')
# f1 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/pruebas algoritmos/concatenar/tiles_puebas/Image 4.lsm')
im = tifffile.imread(f1)

# Phasor tile
dc, g, s, md, ph = phlib.phasor_tile(im, 1024, 1024)
dc1 = phlib.concatenate(g, 3, 4)

plt.figure(1)
plt.imshow(dc1, cmap='gray')
plt.show()
