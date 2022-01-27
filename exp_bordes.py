import numpy as np
import funciones
import tifffile
import matplotlib.pyplot as plt
import mpl_scatter_density

#  voy a importar las 4 imagenes y luego calcular G y S de una sola, de dos promediddas y de 4 promediadas.
f1 = str('/home/bruno/Documentos/TESIS/experimento_bordes/test_2mean(1).lsm')
im1 = tifffile.imread(f1)
f2 = str('/home/bruno/Documentos/TESIS/experimento_bordes/test_2mean(2).lsm')
im2 = tifffile.imread(f2)
f3 = str('/home/bruno/Documentos/TESIS/experimento_bordes/test_2mean(3).lsm')
im3 = tifffile.imread(f3)
f4 = str('/home/bruno/Documentos/TESIS/experimento_bordes/test_2mean(4).lsm')
im4 = tifffile.imread(f4)

g1, s1 = funciones.phasor(im1)  # adquiero el g y s de caso 1
g2, s2 = funciones.phasor(im2)
g3, s3 = funciones.phasor(im3)
g4, s4 = funciones.phasor(im4)

g_avg = (g1 + g2 + g3 + g4) / 4
s_avg = (s1 + s2 + s3 + s4) / 4
im = (im1 + im2 + im3 + im4) / 4
g, s = funciones.phasor(im)

plotty = True
if plotty:
    gx = np.concatenate(g_avg)
    gy = np.concatenate(g)
    sx = np.concatenate(s_avg)
    sy = np.concatenate(s)

    fig = plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(gx, gy, 'b.')
    plt.plot(gx, gx, 'r')
    plt.xlabel('G promediado')
    plt.ylabel('G del promedio de imagenes')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(sx, sy, 'b.')
    plt.plot(sx, sx, 'r')
    plt.xlabel('S promediado')
    plt.ylabel('S del promedio de imagenes')
    plt.grid()
    plt.show()

#  hago la distribucion de un pixel para los 4 G y S
ity1 = np.zeros(30)
ity2 = np.zeros(30)
ity3 = np.zeros(30)
ity4 = np.zeros(30)

for i in range(0, 30):
    ity1[i] = im1[i][512][512]
    ity2[i] = im2[i][512][512]
    ity3[i] = im3[i][512][512]
    ity4[i] = im4[i][512][512]

if plotty:
    plt.figure(2)
    plt.plot(ity1)
    plt.plot(ity2)
    plt.plot(ity3)
    plt.plot(ity4)
    plt.ylabel('Intensidad')
    plt.show()

