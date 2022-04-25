import numpy as np
import PhasorLibrary as ph
import tifffile
import matplotlib.pyplot as plt
from tifffile import imwrite, memmap


'''En este codigo voy a probar como llego al phasor de la 2x2, entonces voy a usar una imagen 2x2 y calcular el phasor
de cada canal y luego concatenar g, s, md, ph y dc. Además con esto ya pruebo guardar el ome.tiff comun y el 
comprimido'''


f1 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Experimentos/caso_18370/lsm/18370_SP_Tile_2x2_a.lsm')
im = tifffile.imread(f1)

# Phasor
aux = np.zeros([len(im), 5, 1024, 1024])
for i in range(len(im)):
    aux[i] = ph.phasor(im[i], harmonic=1)

a = np.asarray([aux[0][4], aux[1][4], aux[2][4], aux[3][4]])
aa = ph.concat_d2(a)

plt.figure(1)
plt.imshow(aa)
plt.show()