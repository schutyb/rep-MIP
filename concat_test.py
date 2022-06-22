import numpy as np
import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt


#  Pruebo la funcion concatenar con una imagen de negros y otra de blancos
#  obtengo una imagen estilo damero con grises en la intersección
prueba_damero = True
if prueba_damero:
    im1 = np.zeros([1024, 1024])
    im2 = np.ones([1024, 1024]) * 256
    dc = np.asarray([im1, im2, im1, im2, im2, im1, im2, im1, im1, im2, im1, im2, im2, im1, im2, im1])
    dc = phlib.concatenate(dc, 4, 4, hper=0.05, vper=0.05)

    plt.figure(1)
    plt.imshow(dc, cmap='gray')
    plt.show()

#  Pruebo la alineación con la imagen de la convalaría
#  se observa que hay un corrimiento vertical
#  y se ajusta utilizando otro porcentaje de solapamiento
alineacion = True
if alineacion:
    route = str('/home/bruno/Documentos/Proyectos/TESIS/MIP/data/Concat/')  # set your file route
    fname = str('convalaria_alignment.lsm')
    im = tifffile.imread(route + fname)
    dc0 = phlib.concatenate(im, 2, 2, hper=0.0, vper=0.0)
    dc1 = phlib.concatenate(im, 2, 2, hper=0.05, vper=0.05)
    dc2 = phlib.concatenate(im, 2, 2, hper=0.07, vper=0.05)

    inter = 'none'

    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    ax[0, 0].imshow(dc0[:1200, :1200], cmap='gray', interpolation=inter)
    ax[0, 1].imshow(dc1[:1200, :1200], cmap='gray', interpolation=inter)
    ax[0, 2].imshow(dc2[:1200, :1200], cmap='gray', interpolation=inter)

    ax[0, 0].set_title('Non overlap')
    ax[0, 1].set_title('5% overlap')
    ax[0, 2].set_title('5% vertical and 7% horizontal overlap')

    ax[1, 0].imshow(dc0[850:1100, 100:950], cmap='gray', interpolation=inter)
    ax[1, 1].imshow(dc1[850:1100, 100:950], cmap='gray', interpolation=inter)
    ax[1, 2].imshow(dc2[850:1100, 100:950], cmap='gray', interpolation=inter)

    plt.show()
