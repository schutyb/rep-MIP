
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import colorsys
import os


filename = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/Nevos/prueba.ome.tiff'
im = tifffile.imread(filename)

phase = np.concatenate(im[4])
bins = np.arange(180)
hist = np.histogram(phase, bins)[0]
hist[0] = 0
bins = np.arange(179)
fases = []

for i in range(len(hist)):
    if hist[i] > int(0.01 * max(hist)):
        fases.append(bins[i])

max_ph = max(fases)
min_ph = min(fases)

aux = np.where(im[4] != 0, im[4], 1000)
aux = np.where(aux < min_ph, min_ph - 1, aux)
aux = np.where(aux == 1000, 0, aux)
aux = np.where(aux > max_ph, max_ph + 1, aux)

#  Normalizar los valores entre ph_min y ph_max
n = 1
#  Suponiendo que tengo la fase supongo que es un espacio hsv con s = 1 y v = 1
rgb = np.zeros([aux.shape[0], aux.shape[1], 3])
rgb2 = np.zeros([aux.shape[0], aux.shape[1], 3])
for i in range(aux.shape[0]):
    for j in range(aux.shape[1]):
        if aux[i][j] == 0:
            rgb[i][j][:] = (0, 0, 0)  # negro
        else:
            rgb[i][j][:] = colorsys.hsv_to_rgb((aux[i][j] - (min_ph - 1)) * (1 / (n * abs(max_ph - min_ph))),
                                               im[3][i][j], 1)

            rgb2[i][j][:] = colorsys.hsv_to_rgb((aux[i][j] - (min_ph - 1)) * (1 / (n * abs(max_ph - min_ph))), 1, 1)


plt.figure(1)
plt.imshow(rgb, interpolation='spline16')

plt.figure(2)
plt.imshow(rgb2, interpolation='spline16')
plt.axis('off')
plt.show()

if False:
    path = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/Melanomas/'
    files_names = os.listdir(path)
    for k in range(len(files_names)):
        im = tifffile.imread(path + files_names[k])

        #  voy a usar el histograma para sacar los datos para normalizar el color,
        #  tomo el histograma de la imagen y tomo un rango donde est'en comprendidos la matoria de los valres de intensidad
        #  los borden del phasor no perjudican en la escala de sa forma.
        phase = np.concatenate(im[4])
        bins = np.arange(180)
        hist = np.histogram(phase, bins)[0]
        hist[0] = 0
        bins = np.arange(179)
        fases = []

        for i in range(len(hist)):
            if hist[i] > int(0.01 * max(hist)):
                fases.append(bins[i])

        max_ph = max(fases)
        min_ph = min(fases)

        aux = np.where(im[4] != 0, im[4], 1000)
        aux = np.where(aux < min_ph, min_ph - 1, aux)
        aux = np.where(aux == 1000, 0, aux)
        aux = np.where(aux > max_ph, max_ph + 1, aux)

        #  Normalizar los valores entre ph_min y ph_max
        n = 1
        #  Suponiendo que tengo la fase supongo que es un espacio hsv con s = 1 y v = 1
        rgb = np.zeros([aux.shape[0], aux.shape[1], 3])
        for i in range(aux.shape[0]):
            for j in range(aux.shape[1]):
                if aux[i][j] == 0:
                    rgb[i][j][:] = (0, 0, 0)  # negro
                else:
                    rgb[i][j][:] = colorsys.hsv_to_rgb((aux[i][j] - (min_ph - 1)) * (1 / (n * abs(max_ph - min_ph))), 1, 1)

        muestra = files_names[k][0:5]
        plt.imsave('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/pseudocolor/Melanomas/' + muestra + '.png', rgb)


    ''' plt.figure(1)
    plt.imshow(rgb, interpolation='spline16')
    plt.axis('off')
    plt.show() '''
