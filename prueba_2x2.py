import numpy as np
import funciones
import tifffile
import matplotlib.pyplot as plt
import cv2
from skimage.filters import median
from tifffile import imwrite, memmap

f1 = str('/home/bruno/Documentos/TESIS/caso_prueba/18370_SP_Tile_2x2_a.lsm')
im = tifffile.imread(f1)
img = np.zeros([30, 1997, 1997])

f2 = str('/home/bruno/Documentos/TESIS/codigos/MIP/test_append.ome.tif')
im_ome = tifffile.imread(f2)

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

plotty = False
if plotty:
    plt.figure(i)
    plt.imshow(img[i], cmap='gray')
    plt.show()

# PHASOR
g, s = funciones.phasor(img)

# FILTRADO
i = 0
while i < 2:
    g_filt = median(g)
    s_filt = median(s)
    i = i + 1

bins = np.arange(0, 255, 1)
img_avg = np.mean(img, axis=0)  # average image
hist, _ = np.histogram(img_avg, bins)

############################################################################################
############################################################################################
# OMETIFF

filename = 'test_append.ome.tif'
shape = (3, 2, 2, g.shape[0], g.shape[1])
#dtype = 'uint16'

# create an empty OME-TIFF file
imwrite(filename, shape=shape, metadata={'axes': 'TZCYX'})

# memory map numpy array to data in OME-TIFF file
tzcyx_stack = memmap(filename)

# write data to memory-mapped array

tzcyx_stack[0] = g
tzcyx_stack[1] = s
tzcyx_stack[2] = img_avg

tzcyx_stack.flush()

############################################################################################
############################################################################################

b = False
if b:
    # PARTE INTERACTIVA
    funciones.interactive(img_avg, hist, bins, g_filt, s_filt, 0.1)
