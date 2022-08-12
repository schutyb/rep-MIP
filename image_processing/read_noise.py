import numpy as np
import tifffile
from skimage.filters import gaussian, median
import matplotlib.pyplot as plt
import PhasorLibrary


def psnr(img_optimal, img):
    mse = np.mean((img_optimal - img) ** 2)
    psnr_aux = 10 * np.log10((255 ** 2) / mse)
    return psnr_aux


# leon las imagenes de 16 promediados, formo una matriz con (3, 9, 30, 1024, 1024)
path16 = '/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Image Processing/filtering/16avg/'
im16a = tifffile.imread(path16 + '1_16avg_3x3.lsm')
im16b = tifffile.imread(path16 + '2_16avg_3x3.lsm')
im16c = tifffile.imread(path16 + '3_16avg_3x3.lsm')
im16 = np.array([im16a, im16b, im16c])
# leo las imagenes de 2 promediados
path2 = '/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Image Processing/filtering/2avg/'
im2a = tifffile.imread(path2 + '1_2avg_3x3.lsm')
im2b = tifffile.imread(path2 + '2_2avg_3x3.lsm')
im2c = tifffile.imread(path2 + '3_2avg_3x3.lsm')
im2 = np.array([im2a, im2b, im2c])

# dejo un array de 810 con todas las imagenes
# im2 = np.reshape(im2, (810, 1024, 1024))[:, 60:980, :]
# im16 = np.reshape(im16, (810, 1024, 1024))[:, :920, :]
im2 = np.reshape(im2, (810, 1024, 1024))
im16 = np.reshape(im16, (810, 1024, 1024))

# aca abajo pruebo filtrar con gauss variando el sigma y midiendo la PSNR
# de esa forma obtengo el sigma para el cual la PSNR el mayor y tomo ese
# sigma como referencia
sigma_stat = False
if sigma_stat:
    sigma = np.arange(0.5, 0.7, 0.05)
    val = np.zeros(len(im2))
    sg = np.zeros(len(sigma))
    for i in range(len(sigma)):
        filt = gaussian(im2, sigma=sigma[i])
        for j in range(len(im2)):
            val[j] = psnr(im16[j], filt[j])
        sg[i] = np.mean(val)


plot = False
if plot:
    # el filtrado optimo se da en sg = 0.75
    num_chanel = 702  # elijo este canal porque es el que tiene mayor intensidad en todos los de ese stack
    filt = gaussian(im2[num_chanel], sigma=0.75, preserve_range=True).astype('int')
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im16[num_chanel])
    ax[1].imshow(im2[num_chanel])
    ax[2].imshow(filt)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.show()

calculos = True
if calculos:
    # rangos de interés [690:719] y region de ineterés en la pseudocolor [0:252, 480:885]
    aux16 = PhasorLibrary.phasor(im16[0:29])
    aux2 = PhasorLibrary.phasor(im2[0:29])
    auxg = PhasorLibrary.phasor(gaussian(im2[0:29], sigma=0.75, preserve_range=True).astype('int'))

    # # Umbralizo todos los g y s sacando la intensidad del background menor a 1
    aux2 = np.asarray(aux2)
    aux2[1] = np.where(aux2[0] > 1, aux2[1], np.zeros(aux2[1].shape))
    aux2[2] = np.where(aux2[0] > 1, aux2[2], np.zeros(aux2[2].shape))
    aux16 = np.asarray(aux16)
    aux16[1] = np.where(aux16[0] > 1, aux16[1], np.zeros(aux16[1].shape))
    aux16[2] = np.where(aux16[0] > 1, aux16[2], np.zeros(aux16[2].shape))
    auxg = np.asarray(auxg)
    auxg[1] = np.where(auxg[0] > 1, auxg[1], np.zeros(auxg[1].shape))
    auxg[2] = np.where(auxg[0] > 1, auxg[2], np.zeros(auxg[2].shape))

    # filtro g y s con la mediana en un while para hacerlo varias veces
    i = 0
    auxm = np.copy(aux2)
    while i < 2:
        auxm[1] = median(auxm[1])
        auxm[2] = median(auxm[2])
        i = i + 1
    dc = np.asarray([aux16[0], aux2[0], auxg[0], auxm[0]])
    g = np.asarray([aux16[1], aux2[1], auxg[1], auxm[1]])
    s = np.asarray([aux16[2], aux2[2], auxg[2], auxm[2]])

    PhasorLibrary.phasor_plot(dc, g, s, ic=2*np.ones(4), title=None, xlabel=None, same_phasor=False)

    # calculo las pSNR de los tres casos contra la img de 16 avg
    gpsnr_comun = psnr(aux16[1], aux2[1])
    gpsnr_g = psnr(aux16[1], auxg[1])
    gpsnr_m = psnr(aux16[1], auxm[1])
    spsnr_comun = psnr(aux16[2], aux2[2])
    spsnr_g = psnr(aux16[2], auxg[2])
    spsnr_m = psnr(aux16[2], auxm[2])

    print('PSNR G sin filtro', gpsnr_comun, 'PSNR G mediana', gpsnr_m, 'PSNR G gauss', gpsnr_g)
    print('PSNR S sin filtro', spsnr_comun, 'PSNR S mediana', spsnr_m, 'PSNR S gauss', spsnr_g)

    # Calculo la fase para el nuevo G y S filtrados con la mediana
    auxm[4] = np.angle(auxm[1] + 1j * auxm[2], deg=True)
    auxg[4] = np.angle(auxg[1] + 1j * auxg[2], deg=True)
    aux2[4] = np.angle(aux2[1] + 1j * aux2[2], deg=True)
    aux16[4] = np.angle(aux16[1] + 1j * aux16[2], deg=True)

    rgb16 = PhasorLibrary.colored_image(aux16[4], [80, 120], md=None, mdinterval=None, outlier_cut=True,
                                        color_scale=0.92)
    rgb2 = PhasorLibrary.colored_image(aux2[4], [80, 120], md=None, mdinterval=None, outlier_cut=True,
                                       color_scale=0.92)
    rgbmed = PhasorLibrary.colored_image(auxm[4], [80, 120], md=None, mdinterval=None, outlier_cut=True,
                                         color_scale=0.92)
    rgbgauss = PhasorLibrary.colored_image(auxg[4], [80, 120], md=None, mdinterval=None, outlier_cut=True,
                                           color_scale=0.92)

    # region de interes [0:252, 480:885]
    # para el [60:89] stack [300:690, 130:790]
    fig2, ax2 = plt.subplots(2, 2)
    ax2[0, 0].imshow(rgb16[300:700, 0:800])
    ax2[0, 1].imshow(rgb2[350:750, 0:800])
    ax2[1, 0].imshow(rgbgauss[350:750, 0:800])
    ax2[1, 1].imshow(rgbmed[350:750, 0:800])
    ax2[0, 0].axis('off')
    ax2[0, 1].axis('off')
    ax2[1, 0].axis('off')
    ax2[1, 1].axis('off')
    ax2[0, 0].set_xlabel('16 avg')
    ax2[0, 1].set_xlabel('2 avg')
    ax2[1, 0].set_xlabel('gauss filt')
    ax2[1, 1].set_xlabel('median filt')
    plt.show()
