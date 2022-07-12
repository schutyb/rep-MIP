import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PhasorLibrary as Ph


''' 
    Define the file name and route as string which is the image or image stack we will read with tifffile module.
    If all the lsm files are stored in the same directory just change the fname.
    the im is a numpy.ndarray with the image stack. 
'''

# SET THE CORRECT ROUTE TO YOUR FILE IN frout AND USE THE FILE NAMED: noise_simulation.lsm in fname

froute = str('/home/bruno/Documentos/Proyectos/TESIS/MIP/data/noise simulation/')
fname = str('noise_simulation.lsm')
f = froute + fname
im = tifffile.imread(f)

''' 
    In this first part it is calculated the G and S without noise which will be the gold standard.
    Also we calculate the modulation and phase, md and ph respectably, the average image of the stack
    and the histogram of the average image. 
'''

dc, g_true, s_true, md, ph = Ph.phasor(im, harmonic=1)
bins = np.arange(0, 255, 1)
hist, _ = np.histogram(dc, bins)

'''
    defino cuatro imagenes a partir de una con un 5 % de solapamiento
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

times = 10
pgc = np.zeros(times)
pgf = np.zeros(times)
psc = np.zeros(times)
psf = np.zeros(times)
psnr_gc = np.zeros(times)
psnr_gf = np.zeros(times)
psnr_sc = np.zeros(times)
psnr_sf = np.zeros(times)

k = 0
while k < times:
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
    dc_concat, g_concat, s_concat, _, _ = Ph.phasor(img_concat)

    '''
        En esta parte calculo el g y s por patches. Calculo el gi y si de cada uno de los 4 cuadrantes, promediando 
        las zonas de solapamiento para obtener un solo g y s. 
    '''
    dc1, g1, s1, _, _ = Ph.phasor(im1)
    dc2, g2, s2, _, _ = Ph.phasor(im2)
    dc3, g3, s3, _, _ = Ph.phasor(im3)
    dc4, g4, s4, _, _ = Ph.phasor(im4)

    # concateno y promedio los gi y si
    g_aux = np.asarray([g1, g2, g3, g4])
    s_aux = np.asarray([s1, s2, s3, s4])
    dc_aux = np.asarray([dc1, dc2, dc3, dc4])
    g_fft = Ph.concat_d2(g_aux)
    s_fft = Ph.concat_d2(s_aux)
    dc_fft = Ph.concat_d2(dc_aux)

    '''
        Ahora tengo:
                    g_true y s_true que son los modelos
                    g_concat y s_concat son los obtenidos concatenando los espectrales
                    g_fft y s_fft son los obtenidos por patches
    '''

    # mido el error pixel a pixel como la diferencia
    egc = abs(g_concat - g_true)
    egf = abs(g_fft - g_true)
    esc = abs(s_concat - s_true)
    esf = abs(s_fft - s_true)

    #  PSNR en las intersecciones
    aux10 = np.zeros(egc.shape)
    pot_gc = np.where(egc == egf, aux10, egc)
    pgc[k] = np.sum(pot_gc ** 2) / len(np.where(pot_gc != 0)[0])
    psnr_gc[k] = 10 * np.log10((255 ** 2) / pgc[k])

    aux11 = np.zeros(egc.shape)
    pot_gf = np.where(egc == egf, aux11, egf)
    pgf[k] = np.sum(pot_gf ** 2) / len(np.where(pot_gf != 0)[0])
    psnr_gf[k] = 10 * np.log10((255 ** 2) / pgf[k])

    aux12 = np.zeros(egc.shape)
    pot_sc = np.where(esc == esf, aux12, esc)
    psc[k] = np.sum(pot_sc ** 2) / len(np.where(pot_sc != 0)[0])
    psnr_sc[k] = 10 * np.log10((255 ** 2) / psc[k])

    aux13 = np.zeros(egc.shape)
    pot_sf = np.where(esf == esc, aux13, esf)
    psf[k] = np.sum(pot_sf ** 2) / len(np.where(pot_sf != 0)[0])
    psnr_sf[k] = 10 * np.log10((255 ** 2) / psf[k])

    k = k + 1

fig_dc, axx = plt.subplots(1, 3, figsize=(15, 5))
axx[0].imshow(dc, cmap='gray', interpolation='none')
axx[0].set_title('Original average intensity')
axx[1].imshow(dc_concat, cmap='gray', interpolation='none')
axx[1].set_title('Concatenation')
axx[2].imshow(dc_fft, cmap='gray', interpolation='none')
axx[2].set_title('Patches')

fig_psnr, ax_psnr = plt.subplots(1, 2, figsize=(16, 4))

ax_psnr[0].plot(np.arange(times), psnr_gc, 'r*-', label='Concatenation')
ax_psnr[0].plot(np.arange(times), psnr_gf, 'b*-', label='Patches')
ax_psnr[0].set_title('PSNR: G image')
ax_psnr[0].set_ylabel('dB')
ax_psnr[0].legend(loc=1)
ax_psnr[0].grid()

ax_psnr[1].plot(np.arange(times), psnr_sc, 'r*-', label='Concatenation')
ax_psnr[1].plot(np.arange(times), psnr_sf, 'b*-', label='Patches')
ax_psnr[1].set_title('PSNR: S image')
ax_psnr[1].set_ylabel('dB')
ax_psnr[1].legend(loc=1)
ax_psnr[1].grid()

psnr_gc = np.mean(psnr_gc)
psnr_gf = np.mean(psnr_gf)

psnr_sc = np.mean(psnr_sc)
psnr_sf = np.mean(psnr_sf)

# voy a graficar para visualizar el error en las intersecciones
plot_err = True
if plot_err:
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(egc, cmap='gray', interpolation='None')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(egf, cmap='gray', interpolation='None')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(esc, cmap='gray', interpolation='None')

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')

    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(esf, cmap='gray', interpolation='None')

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im4, cax=cax, orientation='vertical')

    ax1.set_title("G concatenation")
    ax2.set_title("G patches")
    ax3.set_title("S concatenation")
    ax4.set_title("S patches")

    # diferencia del error entre ambos metodos
    e = abs(egc - egf) * 100000
    e = np.where(e < 255, e, 255)
    e = e/np.max(e)

    fig_err = plt.figure(figsize=(6, 6))
    ax_err = fig_err.add_subplot(111)
    im_err = ax_err.imshow(e, cmap='gray', interpolation='None')
    divider = make_axes_locatable(ax_err)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig_err.colorbar(im_err, cax=cax, orientation='vertical')
    ax_err.set_title('Normalized both methods difference')

    # Grafico el error en la intersección para cada método

    err_aux1 = np.zeros(egc.shape)
    err_egc = np.where(egc == egf, err_aux1, egc)
    err_aux2 = np.zeros(egf.shape)
    err_egf = np.where(egc == egf, err_aux2, egf)
    err_aux3 = np.zeros(esc.shape)
    err_esc = np.where(esc == esf, err_aux3, esc)
    err_aux4 = np.zeros(esf.shape)
    err_esf = np.where(esc == esf, err_aux4, esf)

    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 8))
    img1 = ax2[0, 0].imshow(err_egc, cmap='gray', interpolation='none')
    divider = make_axes_locatable(ax2[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig2.colorbar(img1, cax=cax, orientation='vertical')
    ax2[0, 0].set_title('Error through concatenation in G')

    img2 = ax2[0, 1].imshow(err_egf, cmap='gray', interpolation='none')
    divider = make_axes_locatable(ax2[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig2.colorbar(img2, cax=cax, orientation='vertical')
    ax2[0, 1].set_title('Error through patches in G')

    img3 = ax2[1, 0].imshow(err_esc, cmap='gray', interpolation='none')
    divider = make_axes_locatable(ax2[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig2.colorbar(img3, cax=cax, orientation='vertical')
    ax2[1, 0].set_title('Error through concatenation in S')

    img4 = ax2[1, 1].imshow(err_esc, cmap='gray', interpolation='none')
    divider = make_axes_locatable(ax2[1, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig2.colorbar(img4, cax=cax, orientation='vertical')
    ax2[1, 1].set_title('Error through patches in S')


    # PSNR en las intersecciones
    aux10 = np.zeros(egc.shape)
    pot_gc = np.where(egc == egf, aux10, egc)
    pot_gc = np.sum(pot_gc ** 2) / len(np.where(pot_gc != 0)[0])
    psnr_gc_int = 10 * np.log10((255 ** 2) / pot_gc)        # PSNR
    gc_inter = np.where(egc == egf, aux11, g_concat)  # obtengo el g de la zona de solapamiento
    g_gs_inter = np.where(egc == egf, aux11, g_true)  # gold standr en la zona de solapamiento
    s_gs_inter = np.where(egc == egf, aux11, s_true)
    aux11 = np.zeros(egc.shape)
    pot_gf = np.where(egc == egf, aux11, egf)
    pot_gf = np.sum(pot_gf ** 2) / len(np.where(pot_gf != 0)[0])
    psnr_gf_int = 10 * np.log10((255 ** 2) / pot_gf)        # PSNR
    gf_inter = np.where(egc == egf, aux11, g_fft)  # obtengo el g de la zona de solapamiento
    aux12 = np.zeros(egc.shape)
    pot_sc = np.where(esc == esf, aux12, esc)
    pot_sc = np.sum(pot_sc ** 2) / len(np.where(pot_sc != 0)[0])
    psnr_sc_int = 10 * np.log10((255 ** 2) / pot_sc)  # PSNR
    sc_inter = np.where(egc == egf, aux11, s_concat)  # obtengo el s de la zona de solapamiento
    aux13 = np.zeros(egc.shape)
    pot_sf = np.where(esf == esc, aux13, esf)
    pot_sf = np.sum(pot_sf ** 2) / len(np.where(pot_sf != 0)[0])
    psnr_sf_int = 10 * np.log10((255 ** 2) / pot_sf)  # PSNR
    sf_inter = np.where(egc == egf, aux11, s_fft)  # obtengo el s de la zona de solapamiento

    print('--------------------------------------------------------------------------------')
    print('PSNR en las intersecciones para G concat', psnr_gc_int)
    print('PSNR en las intersecciones para G fft', psnr_gf_int)
    print('PSNR en las intersecciones para S concat', psnr_sc_int)
    print('PSNR en las intersecciones para S fft', psnr_sf_int)
    print('--------------------------------------------------------------------------------')

plot_phasor = True  # grafico el phasor de cada caso
if plot_phasor:
    avg = [dc, dc, dc]
    g = [g_true, g_concat, g_fft]
    s = [s_true, s_concat, s_fft]
    icut = [1, 1, 1]
    titles = ['Gold standard', 'Concatenation', 'Patches']
    fig1 = Ph.phasor_plot(avg, g, s, icut, titles)

    # grafico el phasor en las zonas de intersección
    g2 = [g_gs_inter, gc_inter, gc_inter]
    s2 = [s_gs_inter, sc_inter, sf_inter]
    titles = ['Gold standard', 'Concatenation', 'Patches']
    fig2 = Ph.phasor_plot(avg, g2, s2, icut, titles)

plt.show()
