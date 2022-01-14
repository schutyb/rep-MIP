import tifffile
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


f = str('datos/SP_muestra_20226_Tile_7x4_pos_2a.lsm')
# aca debe entrar el nombre del archivo, y sacar el numero de la muestra
# y el tamano del Tile

m = int(f[28])
n = int(f[30])
muestra = int(f[17:22])
i = j = 1024  # deberia ir el tamano de cada bloque,ese dato sale del metadata
num_chanel = 30

# cargamos el archivo a trabajar
im = tifffile.imread(f)
borde = 0
size_v = int(i * m - 2 * m * borde)
size_h = int(j * n - 2 * n * borde)
#  defino los tamanos de la nueva imagen al sacarle 100 px de cada borde

image = np.zeros([num_chanel, size_v, size_h])
#  imagen donde se guarda las imagenes unidas

for ind1 in range(0, num_chanel):
    ind2 = 0
    cont = 1
    aux_concat = []
    control = 0
    while ind2 < n*m:
        print(ind2)
        if (control % 2) == 0:
            st_hconcat_im = cv2.hconcat([im[ind2][ind1], im[ind2 + 1][ind1]])
            ind2 = ind2 + 2
            while ind2 < n * cont:
                st_hconcat_im = cv2.hconcat([st_hconcat_im, im[ind2][ind1]])
                ind2 = ind2 + 1
            aux_concat.append(st_hconcat_im)
            cont = cont + 1
            control = control+1
        elif (cont*n) < 29:
            st_hconcat_im = cv2.hconcat([im[cont*n-1][ind1], im[cont*n - 2][ind1]])
            ind22 = cont*n - 3
            while ind22 > n * cont - n:
                st_hconcat_im = cv2.hconcat([st_hconcat_im, im[ind22][ind1]])
                ind22 = ind22 - 1
                ind2 = ind2 + 1
            aux_concat.append(st_hconcat_im)
            cont = cont + 1
            cntrol = control + 1

    im_new = cv2.vconcat([aux_concat[0], aux_concat[1]])
    for ind3 in range(2, m):
        im_new = cv2.vconcat([im_new, aux_concat[ind3]])

    image[ind1] = im_new


#img = image[1]
#im_ga = np.double(np.array(img))
#im_fin = np.uint8(im_ga)
#res_img = Image.fromarray(im_fin, mode="L")
#res_img.save('datos/prueba1.png')
