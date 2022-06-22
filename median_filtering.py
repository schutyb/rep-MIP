import numpy as np
import tifffile
from skimage.filters import median
import matplotlib.pyplot as plt
import PhasorLibrary as phlib


filt_time = np.arange(6)
arr_g_psrn = []
arr_s_psrn = []
for i in range(3):
    route_16avg = str('/home/bruno/Documentos/Proyectos/TESIS/MIP/data/filtering/')
    file_16avg = route_16avg + str(i + 1) + str('_16avg_3x3.lsm')
    im_16avg = tifffile.imread(file_16avg)
    route_2avg = str('/home/bruno/Documentos/Proyectos/TESIS/MIP/data/filtering/')
    file_2avg = route_2avg + str(i + 1) + str('_2avg_3x3.lsm')
    im_2avg = tifffile.imread(file_2avg)

    for k in range(len(im_16avg)):
        _, g_optimal, s_optimal, _, _ = phlib.phasor(im_16avg[k])
        _, g, s, _, _ = phlib.phasor(im_2avg[k])

        g_psnr = []
        s_psnr = []
        for j in filt_time:
            ind = 0
            while ind < filt_time[j]:
                g = median(g)
                s = median(s)
                ind = ind + 1
            g_psnr.append(phlib.psnr(g_optimal, g))
            s_psnr.append(phlib.psnr(s_optimal, s))
        arr_g_psrn.append(g_psnr)
        arr_s_psrn.append(s_psnr)

arr_g_psrn = np.asarray(arr_g_psrn)
arr_s_psrn = np.asarray(arr_s_psrn)

fig, ax = plt.subplots(2, figsize=(10, 5))
for i in range(len(arr_g_psrn)):
    ax[0].plot(filt_time, arr_g_psrn[i], '-x')
    ax[1].plot(filt_time, arr_s_psrn[i], '-x')

ax[0].set_title('PSNR for G')
ax[1].set_title('PSNR for S')
ax[0].set_ylabel('PSNR[dB]')
ax[1].set_ylabel('PSNR[dB]')
plt.xlabel('Times (0 ; 5)')

prueba = True
if prueba:
    # regiones interesantes img 2 indice 3, 4 y 5
    optimal = 2
    im_2avg = tifffile.imread(str('/home/bruno/Documentos/Proyectos/TESIS/MIP/data/filtering/1_2avg_3x3.lsm'))
    im_16avg = tifffile.imread(str('/home/bruno/Documentos/Proyectos/TESIS/MIP/data/filtering/1_16avg_3x3.lsm'))
    ind = 4
    dc_optimal, g_optimal, s_optimal, _, _ = phlib.phasor(im_16avg[ind])
    dc, g, s, _, _ = phlib.phasor(im_2avg[ind])

    g_filt = np.copy(g)
    s_filt = np.copy(s)
    j = 0
    while j < optimal:
        g_filt = median(g_filt)
        s_filt = median(s_filt)
        j = j + 1

    dc = [dc_optimal, dc, dc]
    g = [g_optimal, g_filt, g]
    s = [s_optimal, s_filt, s]
    ic = [0, 0, 0]
    titles = ['Gold standard', 'Optimal filtered (twice)', 'Experimental acquisition']
    phlib.phasor_plot(dc, g, s, ic, title=titles)
    plt.show()

# muestro como quedan con varias veces filtrados
if True:
    im_2avg = tifffile.imread(str('/home/bruno/Documentos/Proyectos/TESIS/MIP/data/filtering/2_2avg_3x3.lsm'))
    im_16avg = tifffile.imread(str('/home/bruno/Documentos/Proyectos/TESIS/MIP/data/filtering/2_16avg_3x3.lsm'))
    ind = 4
    dc_optimal, g_optimal, s_optimal, _, _ = phlib.phasor(im_16avg[ind])
    dc, g, s, _, _ = phlib.phasor(im_2avg[ind])

    g_filt = np.copy(g)
    s_filt = np.copy(s)

    g_filt2 = []
    g_filt3 = []
    g_filt4 = []
    s_filt2 = []
    s_filt3 = []
    s_filt4 = []

    j = 0
    while j < 4:
        g_filt = median(g_filt)
        s_filt = median(s_filt)
        if j == 1:
            g_filt2 = g_filt
            s_filt2 = s_filt
        if j == 2:
            g_filt3 = g_filt
            s_filt3 = s_filt
        if j == 3:
            g_filt4 = g_filt
            s_filt4 = s_filt
        j = j + 1

    dc = [dc, dc, dc]
    g = [g_filt2, g_filt3, g_filt4]
    s = [s_filt2, s_filt3, s_filt4]
    ic = [1, 1, 1]
    titles = ['Filtered: twice', 'Filtered: 3 times', 'Filtered: 4 times']
    phlib.phasor_plot(dc, g, s, ic, title=titles)
    plt.show()
