import numpy as np
import tifffile
from skimage.filters import gaussian, median
import matplotlib.pyplot as plt
import PhasorLibrary


def psnr(img_optimal, img):
    mse = np.mean((img_optimal - img) ** 2)
    psnr_aux = 10 * np.log10((255 ** 2) / mse)
    return psnr_aux


path = '/home/bruno/Documentos/Proyectos/Tesis/MIP-Data/Melanocytes/Region de interés/nevus 13-8-22/'
im16 = tifffile.imread(path + 'xxx_R2_16avg_zoomx1.lsm')
im2 = tifffile.imread(path + 'xxx_R1_2avg.lsm')

calculos = True
if calculos:
    aux16 = PhasorLibrary.phasor(im16)
    aux2 = PhasorLibrary.phasor(im2)
    auxg = PhasorLibrary.phasor(gaussian(im2, sigma=0.75, preserve_range=True).astype('int'))

    # mediana
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

    rgb16 = PhasorLibrary.colored_image(aux16[4], [80, 120], md=None, mdinterval=None, outlier_cut=True,
                                        color_scale=0.92)
    rgb2 = PhasorLibrary.colored_image(aux2[4], [80, 120], md=None, mdinterval=None, outlier_cut=True,
                                       color_scale=0.92)
    rgbmed = PhasorLibrary.colored_image(auxm[4], [80, 120], md=None, mdinterval=None, outlier_cut=True,
                                         color_scale=0.92)
    rgbgauss = PhasorLibrary.colored_image(auxg[4], [80, 120], md=None, mdinterval=None, outlier_cut=True,
                                           color_scale=0.92)

    fig2, ax2 = plt.subplots(2, 2)
    ax2[0, 0].imshow(rgb16)
    ax2[0, 1].imshow(rgb2)
    ax2[1, 0].imshow(rgbgauss)
    ax2[1, 1].imshow(rgbmed)
    ax2[0, 0].axis('off')
    ax2[0, 1].axis('off')
    ax2[1, 0].axis('off')
    ax2[1, 1].axis('off')
    plt.show()

# im = tifffile.imread(path + '19837_R5_16avg_zoom2x_channel.lsm')
# c = PhasorLibrary.concatenate(im, 2, 2, hper=0.1, vper=0.1)
# plt.imshow(c, cmap='gray')
# plt.show()