import numpy as np
import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt


f1 = str('/home/bruno/Documentos/TESIS/TESIS/estudio del ruido/exp bordes/lsm/exp_1x1_nevo_1.lsm')
# f1 = str('/home/bruno/Documentos/TESIS/TESIS/IMAGENES LSM/2022/MELANOMAS/14736_SP_Tile_3x6.lsm')
# f1 = str('/home/bruno/Documentos/TESIS/TESIS/imagen.lsm')
im = tifffile.imread(f1)

# Phasor tile
dc, g, s, md, ph = phlib.phasor(im)
# dc, g, s, md, ph = phlib.phasor_tile(im, 1024, 1024)
# dc = phlib.concatenate(dc, 6, 3)
# ph = phlib.concatenate(ph, 6, 3)

# coloracion de la imagen
ic = 2  # intensidad de corte para sacar el fondo
# aux = np.where(dc > ic, ph, np.mean(ph))  # pongo la media en en los valores de fondo para sacar el max y min
# maxi = np.max(aux)
# mini = np.min(aux)
aux = np.where(dc > ic, ph, 0)  # vuelvo a dejar en cero el fondo
# dif = round(360 / (maxi-mini))  # calculo el intervalo de grados
# arr = np.arange(int(mini), int(maxi) + dif, dif)  # arreglo para asignar un color a cada segmento de la muestra

ax = np.zeros([dc.shape[0], dc.shape[0], 3])

''' img_new = np.copy(dc)
cmap = plt.cm.gray
norm = plt.Normalize(img_new.min(), img_new.max())
rgba = cmap(norm(img_new))

for i in range(ph.shape[0]):
    for j in range(ph.shape[1]):
        if ic < aux[i, j] < 265:
            rgba[i, j, :3] = 255, 255, 255  # red
        elif 265 < aux[i, j] < 290:
            rgba[i, j, :3] = 128, 128, 255  # green
        elif 290 < aux[i, j] < 360:
            rgba[i, j, :3] = 0, 255, 255  # blue
        else:
            rgba[i, j, :3] = 0, 0, 0

plt.figure(1)
plt.imshow(rgba)
plt.show()

# Set the colors
# rgba[indices1[0], indices1[1], :3] = 1, 0, 0  # blue
# rgba[indices2[0], indices2[1], :3] = 0, 1, 0  # green
# rgba[indices3[0], indices3[1], :3] = 0, 0, 1  # red '''

filename = str('/home/bruno/Documentos/TESIS/lut/3color-BMR.lut')
with open(filename, 'rb') as f:
    lut = bytearray(f.read())
