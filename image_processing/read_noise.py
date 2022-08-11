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

im2 = np.reshape(im2, (810, 1024, 1024))[:, 60:980, :]
im16 = np.reshape(im16, (810, 1024, 1024))[:, :920, :]

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
    num_chanel = 45
    filt = gaussian(im2[num_chanel], sigma=0.75, preserve_range=True).astype('int')

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im16[num_chanel])
    ax[1].imshow(im2[num_chanel])
    ax[2].imshow(filt)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    # p1 = psnr(im16[num_chanel], im2[num_chanel])
    # p2 = psnr(im16[num_chanel], filt)

aux16 = PhasorLibrary.phasor(im16[690:719])
aux2 = PhasorLibrary.phasor(im2[690:719])
auxg = PhasorLibrary.phasor(gaussian(im2[690:719], sigma=0.75, preserve_range=True).astype('int'))

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

rgbmed = PhasorLibrary.colored_image(auxm[4], [80, 120], md=None, mdinterval=None, outlier_cut=True, color_scale=0.92)
rgbgauss = PhasorLibrary.colored_image(auxg[4], [80, 120], md=None, mdinterval=None, outlier_cut=True, color_scale=0.92)

fig2, ax2 = plt.subplots(1, 2)
ax2[0].imshow(rgbmed)
ax2[1].imshow(rgbgauss)
ax2[0].axis('off')
ax2[1].axis('off')

plt.show()
