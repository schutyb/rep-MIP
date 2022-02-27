import numpy as np
import PhasorLibrary
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import median, gaussian

''' 
    Define the file name and route as string which is the image or image stack we will read with tifffile module.
    If all the lsm files are stored in the same directory just change the fname.
    the im is a numpy.ndarray with the image stack. 
'''

froute = str('/home/bruno/Documentos/TESIS/TESIS/Experimentos/exp_bordes/img_1x1/lsm/')
fname = str('exp_1x1_melanoma_1')
f = froute + fname + str('.lsm')
im = tifffile.imread(f)

''' 
    In this first part it is calculated the G and S without noise which will be the gold standard.
    Also we calculate the modulation and phase, md and ph respectably, the average image of the stack
    and the histogram of the average image. 
'''

g_true, s_true, md, ph = PhasorLibrary.phasor(im)
bins = np.arange(0, 255, 1)
im_avg = np.mean(im, axis=0)  # get the average image
hist, _ = np.histogram(im_avg, bins)

# if you use a new lsm file you should create new ome.tiff file to store the new data set create_file to True
create_file = False
if create_file:
    filename = str('/home/bruno/Documentos/TESIS/TESIS/Experimentos/exp_bordes/img_1x1/omefile/') + fname + \
               str('.ome.tiff')
    data_fft = [g_true, s_true, im_avg, md, ph]
    PhasorLibrary.generate_file(filename, data_fft)

    f2 = str('/home/bruno/Documentos/TESIS/TESIS/Experimentos/exp_bordes/img_1x1/omefile/exp_1x1_melanoma_1.ome'
             '.tiff')
    # Now you have stored in im2: g, s, avg, md and ph.
    im2 = tifffile.imread(f2)
    g_true = im2[0]
    s_true = im2[1]
    im_avg = im2[2]
    md = im2[3]
    ph = im2[4]

'''
    defino cuatro imagenes a partir de una  con un 5 % de solapamiento
    agrego ruido y luego con esa voy a calcular el phasor y filtrarla
    para ver como difieren teniendo la imagen original y puedo contrastar
'''

l = int(im.shape[1] * 0.05)
m = int(im.shape[1] / 2 + l)
n = int(im.shape[1] / 2 - l)
d = len(im)
im1 = np.zeros([d, m, m])
im2 = np.zeros([d, m, m])
im3 = np.zeros([d, m, m])
im4 = np.zeros([d, m, m])

aux1 = np.zeros([d, m, m])
aux2 = np.zeros([d, m, m])
aux3 = np.zeros([d, m, m])
aux4 = np.zeros([d, m, m])

s = len(im[0])
t = int(s / 2)

# Corto las imagenes y agrego ruido asi tengo las 4 imagenes im1 a im4 que luego concateno
for i in range(0, d):
    aux1[i] = im[i][0:m, 0:m]
    im1[i] = aux1[i] + np.random.normal(aux1[i], scale=1.0)

    aux2[i] = im[i][0:m, n:1024]
    im2[i] = aux2[i] + np.random.normal(aux2[i], scale=1.0)

    aux3[i] = im[i][n:1024, 0:m]
    im3[i] = aux3[i] + np.random.normal(aux3[i], scale=1.0)

    aux4[i] = im[i][n:1024, n:1024]
    im4[i] = aux4[i] + np.random.normal(aux4[i], scale=1.0)

# Concateno las imagenes im1 a im4 para luego hacer el phasor
im_aux = np.asarray([im1, im2, im3, im4])
img_concat = PhasorLibrary.concat_d2(im_aux)

# Adquiero el g y el s de la imagen concatenada antes
g_concat, s_concat, _, _ = PhasorLibrary.phasor(img_concat)

'''
    En esta parte calculo el g y s por patches. Calculo el gi y si de cada uno de los 4 cuadrantes, promediando 
    las zonas de solapamiento para obtener un solo g y s. 
'''
g1, s1, _, _ = PhasorLibrary.phasor(im1)
g2, s2, _, _ = PhasorLibrary.phasor(im2)
g3, s3, _, _ = PhasorLibrary.phasor(im3)
g4, s4, _, _ = PhasorLibrary.phasor(im4)

# concateno y promedio los gi y si
g_aux = np.asarray([g1, g2, g3, g4])
g_fft = PhasorLibrary.concat_d2(g_aux)

s_aux = np.asarray([s1, s2, s3, s4])
s_fft = PhasorLibrary.concat_d2(s_aux)

'''
    Ahora tengo:
                g_true y s_true que son los modelos
                g_concat y s_concat son los obtenidos concatenando los espectrales
                g_fft y s_fft son los obtenidos por patches
'''

# TODO ver como comparar los g y s por ahora solo grafico
plotty = False  # grafico g vs g_concat y s vs s_concat
if plotty:
    gx = np.concatenate(g_true)
    gy = np.concatenate(g_concat)
    gy_fft = np.concatenate(g_fft)

    sx = np.concatenate(s_true)
    sy = np.concatenate(s_concat)
    sy_fft = np.concatenate(s_fft)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(gx, gy, 'b.')
    axs[0, 0].set_title('G mediante concat')
    axs[0, 1].plot(sx, sy, 'b.')
    axs[0, 1].set_title('S mediante concat')
    axs[1, 0].plot(gx, gy_fft, 'b.')
    axs[1, 0].set_title('G mediante fft')
    axs[1, 1].plot(sx, sy_fft, 'b.')
    axs[1, 1].set_title('S mediante fft')
    plt.show()

# TODO arreglar la funcion de plotar el phasor, ver si puede ser un objeto donde almacene el phasor asi luego
#  grafico todos los que quiero
plot_phasor = False  # grafico el phasor de cada caso
if plot_phasor:
    avg_noisy = np.mean(img_concat, axis=0)
    fig_true1 = PhasorLibrary.phasor_plot(im_avg, g_true, s_true, 5, avg_noisy, g_concat, s_concat, 5)
    fig_true2 = PhasorLibrary.phasor_plot(im_avg, g_true, s_true, 5, avg_noisy, g_fft, s_fft, 5)
    plt.show()
