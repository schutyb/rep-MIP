import numpy as np
import PhasorLibrary
import tifffile
import matplotlib.pyplot as plt
import cv2
from skimage.filters import median
from tifffile import imwrite, memmap

f1 = str('/home/bruno/Documentos/TESIS/codigos/MIP/prueba2_nevo.lsm')
img = tifffile.imread(f1)


# OMETIFF
def generate_file(file, gsa):
    imwrite(file, data=gsa)
    final_data = memmap(file)
    final_data.flush()
    return final_data

'''
# PHASOR
g_aux, s_aux = funciones.phasor(img)

# FILTRADO
filt = True
i = 1
if filt:
    while i < 2:
        g = median(g_aux)
        s = median(s_aux)
        i = i + 1
else:
    g = np.copy(g_aux)
    s = np.copy(s_aux)

bins = np.arange(0, 255, 1)
img_avg = np.mean(img, axis=0)  # average image
hist, _ = np.histogram(img_avg, bins)

filename = 'gs_nevo_1024.ome.tif'
data = [g, s, img_avg]
generate_file(filename, data)
'''

hacer_calculos = False
if hacer_calculos:
    #############   GENERO LA IMAGEN PROMEDIO DE LOS 30 CANALES PROMEDIANDO LOS BORDES #################

    img = np.zeros([30, 1997, 1997])
    # CONCATENACION
    for i in range(0, 30):
        i1 = im[0][i][:973, :973]
        i2 = im[1][i][:973, 51:]
        i3 = im[2][i][51:, :973]
        i4 = im[3][i][51:, 51:]

        # coloco en la imagen las partes que estan bien
        img[i][0:973, 0:973] = i1
        img[i][0:973, 1024:1997] = i2
        img[i][1024:1997, 0:973] = i3
        img[i][1024:1997, 1024:1997] = i4

        # agrego las zonas donde hay que promediar dos
        img[i][0:974, 973:1024] = (im[0][i][0:974, 973:1024] + im[1][i][0:974, 0:51]) / 2
        img[i][1024:1997, 973:1024] = (im[2][i][51:1024, 973:1024] + im[3][i][51:1024, 0:51]) / 2

        img[i][973:1024, 0:973] = (im[0][i][973:1024, 0:973] + im[2][i][0:51, 0:973]) / 2
        img[i][973:1024, 1024:1997] = (im[1][i][973:1024, 51:1024] + im[3][i][0:51, 51:1024]) / 2

        img[i][973:1024, 973:1024] = (im[0][i][973:1024, 973:1024] + im[1][i][973:1024, 973:1024]
                                      + im[2][i][973:1024, 973:1024] + im[3][i][973:1024, 973:1024]) / 4

    ###################################################################################
    ##########################  CONCATENO LA IMAGEN 2X2 Y HAGO G Y S ##################
    ###################################################################################

    phasor1 = False
    if phasor1:
        plotty = False
        if plotty:
            plt.figure(i)
            plt.imshow(img[i], cmap='gray')
            plt.show()

        # PHASOR
        g_aux, s_aux = PhasorLibrary.phasor(img)

        # FILTRADO
        filt = False
        i = 0
        if filt:
            while i < 2:
                g = median(g_aux)
                s = median(s_aux)
                i = i + 1
        else:
            g = np.copy(g_aux)
            s = np.copy(s_aux)

        bins = np.arange(0, 255, 1)
        img_avg = np.mean(img, axis=0)  # average image
        hist, _ = np.histogram(img_avg, bins)

        filename = '18370_final_data_1_c.ome.tif'
        data = [g, s, img_avg]
        generate_file(filename, data)

    ###################################################################################
    #################### HAGO G Y S DE CADA PATCHE Y LUEGO CONCATENO ##################
    ###################################################################################

    phasor2 = False
    if phasor2:
        g1, s1 = PhasorLibrary.phasor(im[0])
        g2, s2 = PhasorLibrary.phasor(im[1])
        g3, s3 = PhasorLibrary.phasor(im[2])
        g4, s4 = PhasorLibrary.phasor(im[3])

        # dimensiones
        s = int(round(im.shape[2] * 1.95))
        l = int(im.shape[2] * 0.05)
        m = int(im.shape[2])
        n = int(im.shape[2] - l)

        # concateno el g y el s
        g_fft = np.zeros([s, s])

        # concateno para armar el g_fft
        g_fft[0:n, 0:n] = g1[0:n, 0:n]
        g_fft[0:n, m:s] = g2[0:n, l:m]
        g_fft[m:s, 0:n] = g3[l:m, 0:n]
        g_fft[m:s, m:s] = g4[l:m, l:m]

        # verticales
        g_fft1 = np.zeros([n, l])
        g_fft2 = np.zeros([n, l])
        # horizontales
        g_fft3 = np.zeros([l, n])
        g_fft4 = np.zeros([l, n])
        g_fft_centro = np.zeros([l, l])

        # promedio las intersecciones y las pego en la imagen final
        # verticales
        g_fft1 = (g1[0:n, n:m] + g2[0:n, 0:l]) / 2
        g_fft2 = (g3[l:m, n:m] + g4[l:m, 0:l]) / 2
        # horizontales
        g_fft3 = (g1[n:m, 0:n] + g3[0:l, 0:n]) / 2
        g_fft4 = (g2[n:m, l:m] + g4[0:l, l:m]) / 2
        # centro
        g_fft_centro = (g1[n:m, n:m] + g2[0:l, n:m] + g3[0:l, n:m] + g4[0:l, 0:l]) / 4

        # agrego estas ultimas partes a la concatenacion
        g_fft[n:m, n:m] = g_fft_centro
        g_fft[0:n, n:m] = g_fft1
        g_fft[m:s, n:m] = g_fft2
        g_fft[n:m, 0:n] = g_fft3
        g_fft[n:m, m:s] = g_fft4

        # concateno el g y el s
        s_fft = np.zeros([s, s])
        # concateno para armar el s_fft
        s_fft[0:n, 0:n] = s1[0:n, 0:n]
        s_fft[0:n, m:s] = s2[0:n, l:m]
        s_fft[m:s, 0:n] = s3[l:m, 0:n]
        s_fft[m:s, m:s] = s4[l:m, l:m]

        # verticales
        s_fft1 = np.zeros([n, l])
        s_fft2 = np.zeros([n, l])
        # horizontales
        s_fft3 = np.zeros([l, n])
        s_fft4 = np.zeros([l, n])
        s_fft_centro = np.zeros([l, l])

        # promedio las intersecciones y las pego en la imagen final
        # verticales
        s_fft1 = (s1[0:n, n:m] + s2[0:n, 0:l]) / 2
        s_fft2 = (s3[l:m, n:m] + s4[l:m, 0:l]) / 2
        # horizontales
        s_fft3 = (s1[n:m, 0:n] + s3[0:l, 0:n]) / 2
        s_fft4 = (s2[n:m, l:m] + s4[0:l, l:m]) / 2
        # centro
        s_fft_centro = (s1[n:m, n:m] + s2[0:l, n:m] + s3[0:l, n:m] + s4[0:l, 0:l]) / 4

        # agrego estas ultimas partes a la concatenacion
        s_fft[n:m, n:m] = s_fft_centro
        s_fft[0:n, n:m] = s_fft1
        s_fft[m:s, n:m] = s_fft2
        s_fft[n:m, 0:n] = s_fft3
        s_fft[n:m, m:s] = s_fft4

        bins = np.arange(0, 255, 1)
        img_avg_fft = np.mean(img, axis=0)  # average image
        hist, _ = np.histogram(img_avg_fft, bins)

        filename = '18370_final_data_2_c.ome.tif'
        data_fft = [g_fft, s_fft, img_avg_fft]
        generate_file(filename, data_fft)

    ############################################################################################
    ############################################################################################

############################################################################################
#   PRUEBA DE LOS PHASORS #
############################################################################################
probar = False
if probar:
    p1 = str('/home/bruno/Documentos/TESIS/caso_prueba/calculados/18370_final_data_1_a.ome.tif')
    p2 = str('/home/bruno/Documentos/TESIS/caso_prueba/calculados/18370_final_data_2_d.ome.tif')
    gsa1 = tifffile.imread(p1)
    gsa2 = tifffile.imread(p2)

    plot_phasor = True  # grafico el phasor de cada caso
    if plot_phasor:
        fig_true1 = PhasorLibrary.phasor_plot(gsa1[2], gsa1[0], gsa1[1], 5, gsa2[2], gsa2[0], gsa2[1], 5)
        plt.show()
