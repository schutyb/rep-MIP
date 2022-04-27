import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
<<<<<<< HEAD
import PhasorPy as Ph

=======
import PhasorLibrary as Ph
import PhasorPy as phasorPy
>>>>>>> 3807f6715f860870b572ec19ee71c7c3f602a400

''' 
    Define the file name and route as string which is the image or image stack we will read with tifffile module.
    If all the lsm files are stored in the same directory just change the fname.
    the im is a numpy.ndarray with the image stack. 
'''

<<<<<<< HEAD
froute = str('/home/bruno/Documentos/TESIS/TESIS/Experimentos/exp_bordes/img_1x1/lsm/')
=======
froute = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Experimentos/exp_bordes/img_1x1/lsm/')
>>>>>>> 3807f6715f860870b572ec19ee71c7c3f602a400
fname = str('exp_1x1_melanoma_1.lsm')
f = froute + fname
im = tifffile.imread(f)

''' 
    In this first part it is calculated the G and S without noise which will be the gold standard.
    Also we calculate the modulation and phase, md and ph respectably, the average image of the stack
    and the histogram of the average image. 
'''

g_true, s_true, md, ph, dc = Ph.phasor(im, harmonic=1)
bins = np.arange(0, 255, 1)
im_avg = np.mean(im, axis=0)  # get the average image
hist, _ = np.histogram(im_avg, bins)

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
    im1[i] = aux1[i] + abs(np.random.normal(aux1[i], scale=1.0))

    aux2[i] = im[i][0:m, n:1024]
    im2[i] = aux2[i] + abs(np.random.normal(aux2[i], scale=1.0))

    aux3[i] = im[i][n:1024, 0:m]
    im3[i] = aux3[i] + abs(np.random.normal(aux3[i], scale=1.0))

    aux4[i] = im[i][n:1024, n:1024]
    im4[i] = aux4[i] + abs(np.random.normal(aux4[i], scale=1.0))

# Concateno las imagenes im1 a im4 para luego hacer el phasor
im_aux = np.asarray([im1, im2, im3, im4])
img_concat = Ph.concat_d2(im_aux)

# Adquiero el g y el s de la imagen concatenada antes
g_concat, s_concat, _, _, _ = Ph.phasor(img_concat, harmonic=1)

'''
    En esta parte calculo el g y s por patches. Calculo el gi y si de cada uno de los 4 cuadrantes, promediando 
    las zonas de solapamiento para obtener un solo g y s. 
'''
g1, s1, _, _, _ = Ph.phasor(im1, harmonic=1)
g2, s2, _, _, _ = Ph.phasor(im2, harmonic=1)
g3, s3, _, _, _ = Ph.phasor(im3, harmonic=1)
g4, s4, _, _, _ = Ph.phasor(im4, harmonic=1)

# concateno y promedio los gi y si
g_aux = np.asarray([g1, g2, g3, g4])
g_fft = Ph.concat_d2(g_aux)
s_aux = np.asarray([s1, s2, s3, s4])
s_fft = Ph.concat_d2(s_aux)

'''
    Ahora tengo:
                g_true y s_true que son los modelos
                g_concat y s_concat son los obtenidos concatenando los espectrales
                g_fft y s_fft son los obtenidos por patches
'''

err = True
if err:
    egc = abs(g_concat - g_true)  # mido el error pixel a pixel como la diferencia
    egf = abs(g_fft - g_true)
    esc = abs(s_concat - s_true)
    esf = abs(s_fft - s_true)

    # voy a graficar para visualizar el error en las intersecciones
    plot_err = True
    if plot_err:
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(egc, interpolation='None')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(egf, interpolation='None')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        ax3 = fig.add_subplot(223)
        im3 = ax3.imshow(esc, interpolation='None')

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')

        ax4 = fig.add_subplot(224)
        im4 = ax4.imshow(esf, interpolation='None')

        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im4, cax=cax, orientation='vertical')

        ax1.set_title("G concatenación")
        ax2.set_title("G fft")
        ax3.set_title("S concatenación")
        ax4.set_title("S fft")

        e = abs(egc - egf) * 100000
        e = np.where(e < 255, e, 255)

        plt.figure(2)
        plt.title('Diferencia del error entre ambos metodos')
        im = plt.imshow(e, cmap='gray')

        #  Potencia del error
        pgc = np.mean(egc ** 2)
        pgf = np.mean(egf ** 2)
        psc = np.mean(esc ** 2)
        psf = np.mean(esf ** 2)

        print('Potencia del error en G mediante concatenacion', pgc)
        print('Potencia del error en G mediante fft', pgf)
        print('Potencia del error en S mediante concatenacion', psc)
        print('Potencia del error en S mediante fft', psf)

        #  potencia del error en las intersecciones
        aux10 = np.zeros(egc.shape)
        pot_gc = np.where(egc == egf, aux10, egc)
        pot_gc = np.sum(pot_gc ** 2) / len(np.where(pot_gc != 0)[0])

        aux11 = np.zeros(egc.shape)
        pot_gf = np.where(egc == egf, aux11, egf)
        pot_gf = np.sum(pot_gf ** 2) / len(np.where(pot_gf != 0)[0])

        aux12 = np.zeros(egc.shape)
        pot_sc = np.where(esc == esf, aux12, esc)
        pot_sc = np.sum(pot_sc ** 2) / len(np.where(pot_sc != 0)[0])

        aux13 = np.zeros(egc.shape)
        pot_sf = np.where(esf == esc, aux13, esf)
        pot_sf = np.sum(pot_sf ** 2) / len(np.where(pot_sf != 0)[0])

        print('------------------------------------------------------------------------------------------')
        print('Potencia del error en las intersecciones para G concat', pot_gc)
        print('Potencia del error en las intersecciones para G fft', pot_gf)
        print('Potencia del error en las intersecciones para S concat', pot_sc)
        print('Potencia del error en las intersecciones para S fft', pot_sf)
        print('------------------------------------------------------------------------------------------')
        print('Diferencia de potencias entre gc y gf', abs(pot_gc - pot_gf))
        print('Diferencia de potencias entre sc y sf', abs(pot_sc - pot_sf))

plot_phasor = True  # grafico el phasor de cada caso
if plot_phasor:
    avg = [im_avg, im_avg, im_avg]
    g = [g_true, g_concat, g_fft]
    s = [s_true, s_concat, s_fft]
    icut = [3, 3, 3]
    titles = ['Gold standar', 'Mediante concatenación', 'Mediante fft']
<<<<<<< HEAD
    fig, _, _ = Ph.phasor_plot(avg, g, s, icut, titles)
=======
    fig, _, _ = phasorPy.phasor_plot(avg, g, s, icut, titles)
>>>>>>> 3807f6715f860870b572ec19ee71c7c3f602a400

plt.show()
