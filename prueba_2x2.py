import numpy as np
import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt
from tifffile import imwrite, memmap
import cv2


'''En este codigo voy a probar como llego al phasor de la 2x2, entonces voy a usar una imagen 2x2 y calcular el phasor
de cada canal y luego concatenar g, s, md, ph y dc. Además con esto ya pruebo guardar el ome.tiff comun y el 
comprimido'''


# f1 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Experimentos/caso_18370/lsm/18370_SP_Tile_2x2_b.lsm')
# f1 = str('/home/bruno/Documentos/TESIS/TESIS/IMAGENES LSM/2022/MELANOMAS/16952_SP_Tile_4x3.lsm')
# f1 = str('/home/bruno/Documentos/TESIS/TESIS/IMAGENES LSM/2022/MELANOMAS/19686_SP_Tile_7x4.lsm')
# f1 = str('/home/bruno/Documentos/TESIS/TESIS/3x3_2avg.lsm')
# im = tifffile.imread(f1)

# Phasor tile
# dc, g, s, md, ph = phlib.phasor_tile(im, 1024, 1024)


def concatenate(im, m, n, per=0.05):
    d = im.shape[1]
    aux = np.zeros([d * m, d * n])  # store the concatenated image

    # Horizontal concatenate
    i = 0
    j = 0
    while j < m * n:
        aux[i * d: i * d + d, 0:d] = im[j][0:, 0:d]  # store the first image horizontally
        k = 1
        acum = 0
        while k < n:
            ind1 = round(((1 - per) + acum) * d)
            ind2 = round(ind1 + per * d)
            ind3 = round(ind2 + (1 - per) * d)
            aux[i * d:i * d + d, ind1:ind2] = (aux[i * d:i * d + d, ind1:ind2] + im[j + k][0:, 0:round(per * d)]) / 2
            aux[i * d:i * d + d, ind2:ind3] = im[j + k][0:, round(per * d):d]
            acum = (1 - per) + acum
            k = k + 1
        i = i + 1
        j = j + n

    # Vertical concatenate
    img = np.zeros([round(d * (m - per * (m - 1))), round(d * (n - per * (n - 1)))])
    img[0:d, 0:] = aux[0:d, 0:img.shape[1]]
    k = 1
    acum = 0
    while k < m:
        ind1 = round(((1 - per) + acum) * d)
        ind2 = round(ind1 + per * d)
        ind3 = round(ind2 + per * d)
        ind4 = round(ind3 + (1 - per) * d)

        img[ind1:ind2, 0:] = (aux[ind1:ind2, 0:img.shape[1]] + aux[ind2:ind3, 0:img.shape[1]]) / 2
        img[ind3:ind4, 0:] = aux[ind3:ind4, 0:img.shape[1]]
        acum = (1 - per) + acum
        k = k + 1

    return img


i1 = cv2.imread('/home/bruno/Documentos/TESIS/gris.png', cv2.IMREAD_GRAYSCALE)
i2 = cv2.imread('/home/bruno/Documentos/TESIS/negro.png', cv2.IMREAD_GRAYSCALE)
dc = np.asarray([i1, i2, i1, i2, i2, i1, i2, i1, i1, i2, i1, i2, i2, i1, i2, i1])

dc1 = concatenate(dc, 4, 4)
# g1 = concatenate(g, 3, 4)
# s1 = concatenate(s, 3, 4)

plt.figure(1)
plt.imshow(dc1, cmap='gray')
plt.show()
