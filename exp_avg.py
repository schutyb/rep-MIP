import numpy as np
<<<<<<< HEAD
import PhasorLibrary
=======
import PhasorLibrary as Ph
>>>>>>> 3807f6715f860870b572ec19ee71c7c3f602a400
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''
En este codigo utilizo imagenes con diferentes promedios para comparar
calculo los g y s con 16 promedios y los tomo como gold standr y luego calculo con los demas promedios 
'''

#  The 16 averages is the gold standar
<<<<<<< HEAD
file = str('/home/bruno/Documentos/TESIS/TESIS/Experimentos/exp_avg/test_16_mean.lsm')
gs = tifffile.imread(file)
g, s, _, _, dc = PhasorLibrary.phasor(gs, harmonic=1)


fig = PhasorLibrary.interactive(dc, g, s, 0.2)

=======
file = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Experimentos/exp_avg/test_16_mean.lsm')
im = tifffile.imread(file)
g, s, _, _, dc = Ph.phasor(im, harmonic=1)

l = int(im.shape[1] * 0.05)
m = int(im.shape[1] / 2 + l)
n = int(im.shape[1] / 2 - l)
d = len(im)

for i in range(4):
    file = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Experimentos/exp_avg/2_avg/test_2mean_') + str(i + 1) + \
           str('.lsm')
    aux = tifffile.imread(file)
    if i == 0:
        aux1 = aux[0:30, 0:m, 0:m]
        g1, s1, _, _, dc1 = Ph.phasor(aux1, harmonic=1)
    elif i == 1:
        aux2 = aux[0:30, 0:m, n:1024]
        g2, s2, _, _, dc2 = Ph.phasor(aux2, harmonic=1)
    elif i == 2:
        aux3 = aux[0:30, n:1024, 0:m]
        g3, s3, _, _, dc3 = Ph.phasor(aux3, harmonic=1)
    elif i == 3:
        aux4 = aux[0:30, n:1024, n:1024]
        g4, s4, _, _, dc4 = Ph.phasor(aux4, harmonic=1)

aux_s = np.asarray([s1, s2, s3, s4])
sn = Ph.concat_d2(aux_s)

aux_g = np.asarray([g1, g2, g3, g4])
gn = Ph.concat_d2(aux_g)

aux_dc = np.asarray([dc1, dc2, dc3, dc4])
dcn = Ph.concat_d2(aux_dc)

err = True
if err:
    eg = abs(g - gn)  # mido el error pixel a pixel como la diferencia
    es = abs(s - sn)
    edc = abs(dc - dcn)

    # voy a graficar para visualizar el error en las intersecciones
    plot_err = True
    if plot_err:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(eg, interpolation='None')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(es, interpolation='None')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(edc, interpolation='None')

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')

        ax1.set_title("g")
        ax2.set_title("s")
        ax3.set_title("dc")

        pg = np.mean(eg ** 2)
        ps = np.mean(es ** 2)
        pdc = np.mean(edc ** 2)
        print('Potencia del error en g', pg)
        print('Potencia del error en s', ps)
        print('Potencia del error en dc', pdc)


plot_phasor = True  # grafico el phasor de cada caso
if plot_phasor:
    avg = [dc, dcn]
    gl = [g, gn]
    sl = [s, sn]
    icut = [5, 5]
    titles = ['Gold standar', 'Experimental']
    fig = Ph.phasor_plot(avg, gl, sl, icut, titles)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121)
im1 = ax1.imshow(dc, interpolation='None')
ax2 = fig.add_subplot(122)
im2 = ax2.imshow(dcn, interpolation='None')
ax1.set_title("Gold standar Intensity")
ax2.set_title("Experimental Intensity")

plt.show()
>>>>>>> 3807f6715f860870b572ec19ee71c7c3f602a400
