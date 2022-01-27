import numpy as np
import funciones
import tifffile
import matplotlib.pyplot as plt
import mpl_scatter_density

f1 = str('/home/bruno/Documentos/TESIS/experimento_bordes/img_1x1/prueba2_nevo.lsm')
im = tifffile.imread(f1)
g_true, s_true = funciones.phasor(im)

#  defino cuatro imagenes a partir de una  con un 5 % de solapamiento
#  agrego ruido y luego con esa voy a calcular el phasor y filtrarla
#  para ver como difieren tengo la imagen original y puedo contrastar

# el tamano de las imagenes es la mitad mas el 5%
l = int(im.shape[1] * 0.05)
m = int(im.shape[1] / 2 + l)
n = int(im.shape[1] / 2 - l)

im1 = np.zeros([len(im), m, m])
im2 = np.zeros([len(im), m, m])
im3 = np.zeros([len(im), m, m])
im4 = np.zeros([len(im), m, m])

aux1 = np.zeros([len(im), m, m])
aux2 = np.zeros([len(im), m, m])
aux3 = np.zeros([len(im), m, m])
aux4 = np.zeros([len(im), m, m])

s = len(im[0])
t = int(s/2)
img_concat = np.zeros([len(im), s, s])

# verticales
imp1 = np.zeros([len(im[0]), n, 2 * l])
imp2 = np.zeros([len(im[0]), n, 2 * l])
# horizontales
imp3 = np.zeros([len(im[0]), 2 * l, n])
imp4 = np.zeros([len(im[0]), 2 * l, n])
imp_centro = np.zeros([len(im[0]), 2*l, 2*l])

# corto las imagenes y agrego ruido
for i in range(0, len(im)):
    aux1[i] = im[i][0:m, 0:m]
    im1[i] = aux1[1] + np.random.normal(aux1[i])

    aux2[i] = im[i][0:m, n:1024]
    im2[i] = aux2[1] + np.random.normal(aux2[i])

    aux3[i] = im[i][n:1024, 0:m]
    im3[i] = aux3[1] + np.random.normal(aux3[i])

    aux4[i] = im[i][n:1024, n:1024]
    im4[i] = aux4[1] + np.random.normal(aux4[i])

    # concateno la imagen para volver a armarla y hacer el phasor
    img_concat[i][0:n, 0:n] = im1[i][0:n, 0:n]
    img_concat[i][0:n, m:s] = im2[i][0:n, 2*l:m]
    img_concat[i][m:s, 0:n] = im3[i][2*l:m, 0:n]
    img_concat[i][m:s, m:s] = im4[i][2*l:m, 2*l:m]

    # promedio las intersecciones y las pego en la imagen final

    # verticales
    imp1[i] = (im1[i][0:n, n:m] + im2[i][0:n, 0:2*l]) / 2
    imp2[i] = (im3[i][2*l:m, n:m] + im4[i][2*l:m, 0:2*l]) / 2
    # horizontales
    imp3[i] = (im1[i][n:m, 0:n] + im3[i][0:2*l, 0:n]) / 2
    imp4[i] = (im2[i][n:m, 2*l:m] + im4[i][0:2*l, 2*l:m]) / 2
    # centro
    imp_centro[i] = (im1[i][n:m, n:m] + im2[i][0:2*l, n:m] + im3[i][0:2*l, n:m] + im4[i][0:2*l, 0:2*l]) / 4

    # agrego estas ultimas partes a la concatenacion
    img_concat[i][n:m, n:m] = imp_centro[i]
    img_concat[i][0:n, n:m] = imp1[i]
    img_concat[i][m:s, n:m] = imp2[i]
    img_concat[i][n:m, 0:n] = imp3[i]
    img_concat[i][n:m, m:s] = imp4[i]


# adquiero el g y el s
gnew, snew = funciones.phasor(img_concat)

plot_all = False
if plot_all:
    plotty = True  # grafico g vs gnew y s vs snew
    if plotty:
        gx = np.concatenate(gnew)
        gy = np.concatenate(g_true)
        sx = np.concatenate(snew)
        sy = np.concatenate(s_true)

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
        # plt.show()

    plott = True  # grafico las imagenes de g obtenidas
    if plott:
        plt.figure(2)
        plt.imshow(g_true, cmap='gray')
        plt.figure(3)
        plt.imshow(gnew, cmap='gray')
        plt.show()

bins = np.arange(0, 255, 1)
img_avg_true = np.mean(im, axis=0)
hist_true, _ = np.histogram(img_avg_true, bins)

img_avg_new = np.mean(img_concat, axis=0)
hist_new, _ = np.histogram(img_avg_new, bins)

plot_avg = True
if plot_avg:
    fig2 = plt.figure(4)
    plt.subplot(1, 2, 1)
    plt.imshow(img_avg_true, cmap='gray')
    plt.xlabel('Img promedio original')

    plt.subplot(1, 2, 2)
    plt.imshow(img_avg_new, cmap='gray')
    plt.xlabel('Img promedio nueva')

    plt.show()

plot_phasor = True  # grafico el phasor de cada caso
if plot_phasor:
    out1 = funciones.interactive(img_avg_true, hist_true, bins, g_true, s_true, 0.1)
    out2 = funciones.interactive(img_avg_new, hist_new, bins, gnew, snew, 0.1)
