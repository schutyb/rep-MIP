import numpy as np
import tifffile
from skimage.filters import gaussian
import matplotlib.pyplot as plt


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

# el filtrado optimo se da en sg = 0.75
num_chanel = 22
filt = gaussian(im2[num_chanel], sigma=0.75, preserve_range=True).astype('int')

plot = True
if plot:
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im16[num_chanel])
    ax[1].imshow(im2[num_chanel])
    ax[2].imshow(filt)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.show()

p1 = psnr(im16[num_chanel], im2[num_chanel])
p2 = psnr(im16[num_chanel], filt)
