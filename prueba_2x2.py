import numpy as np
import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt
from tifffile import imwrite, memmap

'''En este codigo voy a probar como llego al phasor de la 2x2, entonces voy a usar una imagen 2x2 y calcular el phasor
de cada canal y luego concatenar g, s, md, ph y dc. Además con esto ya pruebo guardar el ome.tiff comun y el 
comprimido'''

f1 = str('/home/bruno/Documentos/TESIS/TESIS/Experimentos/caso_18370/18370/18370_SP_Tile_2x2_b.lsm')
f1 = str('/home/bruno/Documentos/TESIS/TESIS/IMAGENES LSM/2022/NEVOS/15410_SP_Tile_4x4.lsm')
# f1 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/IMAGENES LSM/2022/MELANOMAS/16400_SP_Tile_12x8.lsm')
im = tifffile.imread(f1)

# Phasor tile
dc, g, s, md, ph = phlib.phasor_tile(im, 1024, 1024)


def concatenate(im, m, n):
    d = im.shape[2]
    aux = np.zeros([m * d, n * d])  # store the concatenated image

    i = 0
    while i < m:

        # In this part concatenate the first two images of every raw in the tile image
        aux_mean = (im[i][0:, int(0.95 * d):d] + im[i + 1][0:, 0:int(0.05 * d + 1)]) / 2
        aux[i * d: i * d + d, 0:int(0.95 * d)] = im[i][0:, 0:int(0.95 * d)]
        aux[i * d: i * d + d, int(0.95 * d):d] = aux_mean
        aux[i * d: i * d + d, d:2 * d - int(0.05 * d)] = im[i + 1][0:, int(0.05 * d):d]

        # here it is concatenate the rest of the raw
        cont = 2
        while cont < n:
            aux_mean = (aux[i * d: i * d + d, d * cont - int(0.05 * d):d * cont] + im[i + cont][0:, 0:int(
                0.05 * d)]) / 2
            aux[i * d: i * d + d, d * cont - int(0.05 * d):d * cont] = aux_mean
            aux[i * d: i * d + d, d * cont:d * (cont + 1)] = im[i + cont][0:, int(0.05 * d):d]
            cont = cont + 1

        i = i + 1

    # cuando termina este for concateno hacia abajo
    return aux


aux = concatenate(dc, 4, 4)
